import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import penman
import torch
from smatchpp import eval_statistics, preprocess, solvers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, HfArgumentParser, PreTrainedModel

from multi_amr.evaluate.backoff_smatch import BackOffSmatchpp
from multi_amr.tokenization import AMRTokenizerWrapper, TokenizerType


def batch_translate(
    texts: List[str], src_lang: str, model: PreTrainedModel, tok_wrapper: AMRTokenizerWrapper, **gen_kwargs
) -> Dict[str, list]:
    """Translates a given text of a given source language with a given model and tokenizer. The generation is guided by
    potential keyword-arguments, which can include arguments such as max length, logits processors, etc.
    :param texts: batch of texts to translate (must be in same language)
    :param src_lang: source language
    :param model: AMR finetuned model
    :param tok_wrapper: tokenizer wrapper
    :param gen_kwargs: potential keyword arguments for the generation process
    :return: the translation (linearized AMR graph)
    """
    if isinstance(texts, str):
        raise ValueError("Expected 'texts' to be a list of strings")

    task_prefix = ""
    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB):
        if src_lang not in tok_wrapper.tokenizer.lang_code_to_id:
            raise KeyError(
                f"src_lang {src_lang} not supported by this tokenizer of type"
                f" {tok_wrapper.tokenizer_type}. Valid src_langs are"
                f" {', '.join(tok_wrapper.tokenizer.lang_code_to_id.keys())}"
            )
        # Set the source lang to the main language so that the correct token can be added (not used by T5)
        tok_wrapper.tokenizer.src_lang = src_lang
        gen_kwargs["decoder_start_token_id"] = tok_wrapper.amr_token_id
    elif tok_wrapper.tokenizer_type in (TokenizerType.T5, TokenizerType.BLOOM):
        # T5 can use prefixes
        task_prefix = f"translate {src_lang} to {tok_wrapper.amr_token}: "

    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.BART, TokenizerType.NLLB, TokenizerType.T5):
        # ENCODER-DECODERS
        encoded = tok_wrapper(
            [task_prefix + t for t in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(model.device)
    else:
        # DECODERS-ONLY
        raise NotImplementedError

    # Make sure that we return all the beams
    beam_size = gen_kwargs["num_beams"]
    gen_kwargs["num_return_sequences"] = beam_size

    with torch.inference_mode():
        generated = model.generate(**encoded, output_scores=True, return_dict_in_generate=True, **gen_kwargs)

    generated["sequences"] = generated["sequences"].cpu()
    generated["sequences_scores"] = generated["sequences_scores"].cpu()
    best_scoring_results = {"graph": [], "status": []}

    if len(generated["sequences_scores"]) != beam_size * len(texts):
        raise ValueError(
            f"Expected {beam_size * len(texts)} sequences after beam search (beam_size * batch_size),"
            f" but got {len(generated['sequences_scores'])}"
        )

    # Select the best item from the beam: the sequence with best status and highest score
    for sample_idx in range(0, len(generated["sequences_scores"]), beam_size):
        sequences = generated["sequences"][sample_idx : sample_idx + beam_size]
        scores = generated["sequences_scores"][sample_idx : sample_idx + beam_size].tolist()
        outputs = tok_wrapper.batch_decode_amr_ids(sequences)
        statuses = outputs["status"]
        graphs = outputs["graph"]
        zipped = zip(statuses, scores, graphs)
        # Lowest status first (OK=0, FIXED=1, BACKOFF=2), highest score second
        best = sorted(zipped, key=lambda item: (item[0].value, -item[1]))[0]
        best_scoring_results["graph"].append(best[2])
        best_scoring_results["status"].append(best[0])

    # Returns dictionary with "graph" and "status" keys
    return best_scoring_results


def get_resources(model_name_or_path: str) -> Tuple[PreTrainedModel, AMRTokenizerWrapper]:
    """Get the relevant model, tokenizer and logits_processor. The loaded model depends on whether the multilingual
    model is requested, or not. If not, an English-only model is loaded. The model can be optionally quantized
    for better performance.
    :param model_name_or_path: name or path of model and tokenizer
    :return: the loaded model, tokenizer, and logits processor
    """
    tok_wrapper = AMRTokenizerWrapper.from_pretrained(model_name_or_path)
    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.T5):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model = model.cuda()
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tok_wrapper.tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tok_wrapper.tokenizer))

    model.eval()

    return model, tok_wrapper


def evaluate(
    model_name: str,
    dref: str,
    dataset_name: str,
    src_lang: str,
    batch_size: int = 8,
    num_beams: int = 5,
    max_new_tokens: int = 900,
):
    model, tok_wrapper = get_resources(model_name)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
    }

    pdin = Path(dref).resolve()
    results = {
        "sentence": [],
        "fname": [],
        "sentid": [],
        "uid": [],
        "penman_str": [],
        "status": [],
        "reference_penman_str": [],
    }
    batch = {
        "sentence": [],
        "fname": [],
        "sentid": [],
        "uid": [],
        "penman_str": [],
        "status": [],
        "reference_penman_str": [],
    }

    # Dirty way to get total number of graphs (double iteration = ew)
    num_graphs = 0
    for pfin in pdin.rglob("*.txt"):
        with pfin.open(encoding="utf-8") as fhin:
            for _ in penman.iterdecode(fhin):
                num_graphs += 1

    pbar = tqdm(unit="graph", total=num_graphs)
    for pfin in pdin.rglob("*.txt"):
        with pfin.open(encoding="utf-8") as fhin:
            for graph in penman.iterdecode(fhin):
                sentid = graph.metadata["id"]
                sentence = graph.metadata["snt"]
                fname = pfin.name
                uid = f"{fname}__{sentid}"
                batch["sentence"].append(sentence)
                batch["sentid"].append(sentid)
                batch["fname"].append(fname)
                batch["uid"].append(uid)
                graph.metadata = {}
                batch["reference_penman_str"].append(penman.encode(graph))

                if len(batch["sentence"]) == batch_size:
                    outputs = batch_translate(
                        batch["sentence"], src_lang=src_lang, model=model, tok_wrapper=tok_wrapper, **gen_kwargs
                    )
                    results["penman_str"].extend([penman.encode(graph) for graph in outputs["graph"]])
                    results["status"].extend([status.name.lower() for status in outputs["status"]])

                    results["sentence"].extend(batch["sentence"])
                    results["reference_penman_str"].extend(batch["reference_penman_str"])
                    results["sentid"].extend(batch["sentid"])
                    results["fname"].extend(batch["fname"])
                    results["uid"].extend(batch["uid"])
                    batch = {
                        "sentence": [],
                        "fname": [],
                        "sentid": [],
                        "uid": [],
                        "penman_str": [],
                        "status": [],
                        "reference_penman_str": [],
                    }
                    pbar.update(batch_size)

    # Rest
    if len(batch["sentence"]):
        outputs = batch_translate(
            batch["sentence"], src_lang=src_lang, model=model, tok_wrapper=tok_wrapper, **gen_kwargs
        )
        results["penman_str"].extend([penman.encode(graph) for graph in outputs["graph"]])
        results["status"].extend([status.name.lower() for status in outputs["status"]])

        results["sentence"].extend(batch["sentence"])
        results["reference_penman_str"].extend(batch["reference_penman_str"])
        results["sentid"].extend(batch["sentid"])
        results["fname"].extend(batch["fname"])
        results["uid"].extend(batch["uid"])
        pbar.update(len(batch["sentence"]))
    pbar.close()

    graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
    ilp = solvers.ILP()
    printer = eval_statistics.ResultPrinter(score_type="micro", do_bootstrap=True, output_format="json")
    smatch_metric = BackOffSmatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer, printer=printer)

    score, optimization_status, _ = smatch_metric.score_corpus(results["reference_penman_str"], results["penman_str"])
    score = score["main"]
    score = {
        "smatch_precision": score["Precision"],
        "smatch_recall": score["Recall"],
        "smatch_fscore": score["F1"],
    }
    print(score)
    print(f"SMATCH evaluation scores of {model_name}")

    pfmodel = Path(model_name)
    if pfmodel.exists() and pfmodel.is_dir():
        pfmodel.joinpath(f"{dataset_name}_results.json").write_text(json.dumps(score, indent=4), encoding="utf-8")

        pfpreds = pfmodel.joinpath(f"{dataset_name}_predictions.tsv")
        df = pd.DataFrame(results)
        df.to_csv(pfpreds, sep="\t", encoding="utf-8", index=False)

        pfpreds = pfpreds.with_stem(f"{pfpreds.stem}-preds-only").with_suffix(".txt")
        pfrefs = pfpreds.with_stem(f"{pfpreds.stem}-refs-only").with_suffix(".txt")
        predictions = df["penman_str"].tolist()
        pfpreds.write_text("\n\n".join(predictions))
        references = df["reference_penman_str"].tolist()
        pfrefs.write_text("\n\n".join(references))


@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "the directory containing the model to evaluate"})
    dref: str = field(metadata={"help": "Directory with references (amr/test files)"})
    dataset_name: str = field(metadata={"help": "Will be used in filenames when writing output files."})
    src_lang: str = field(metadata={"help": "which source language to use. This can be a language code"})
    batch_size: Optional[int] = field(
        default=8, metadata={"help": "batch size (lower this if you get out-of-memory errors)"}
    )
    num_beams: Optional[int] = field(default=5, metadata={"help": "number of beams for generation"})
    max_new_tokens: Optional[int] = field(default=900, metadata={"help": "max. new tokens to generate"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    evaluate(
        model_name=script_args.model_name,
        dref=script_args.dref,
        dataset_name=script_args.dataset_name,
        src_lang=script_args.src_lang,
        batch_size=script_args.batch_size,
        num_beams=script_args.num_beams,
        max_new_tokens=script_args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
