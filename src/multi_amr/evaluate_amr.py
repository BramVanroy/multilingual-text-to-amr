from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from datasets import DatasetDict
from multi_amr.constraints import AMRLogitsProcessor
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from smatchpp import Smatchpp, preprocess, solvers
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    LogitsProcessorList,
    PreTrainedModel,
)


def batch_translate(
    texts: List[str], src_lang: str, model: PreTrainedModel, tok_wrapper: AMRTokenizerWrapper, **gen_kwargs
) -> List[str]:
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
    elif tok_wrapper.tokenizer_type in (TokenizerType.T5, TokenizerType.BLOOM):
        # T5 can use prefixes
        task_prefix = f"translate {src_lang} to {AMR_LANG_CODE}: "

    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.T5):
        # Task prefix empty for MBART and NLLB but not for T5 (cf. above)
        texts = [task_prefix + text for text in texts]
        encoded = tok_wrapper(texts, return_tensors="pt", padding=True)
    else:
        texts = [task_prefix + text + "\n" + tok_wrapper.tokenizer.eos_token for text in texts]
        encoded = tok_wrapper(texts, return_tensors="pt", padding=True)

    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        generated = model.generate(**encoded, **gen_kwargs).cpu()
    return tok_wrapper.decode_and_fix_amr(generated)


def get_resources(model_name_or_path: str) -> Tuple[PreTrainedModel, AMRTokenizerWrapper, AMRLogitsProcessor]:
    """Get the relevant model, tokenizer and logits_processor. The loaded model depends on whether the multilingual
    model is requested, or not. If not, an English-only model is loaded. The model can be optionally quantized
    for better performance.
    :param model_name_or_path: name or path of model and tokenizer
    :return: the loaded model, tokenizer, and logits processor
    """
    tok_wrapper = AMRTokenizerWrapper.from_pretrained(model_name_or_path)
    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.T5):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

    model.resize_token_embeddings(len(tok_wrapper.tokenizer))
    model.eval()
    logits_processor = AMRLogitsProcessor(tok_wrapper, model.config.max_length)

    return model, tok_wrapper, logits_processor


def batchify(sentences: List[str], batch_size: int = 8):
    """Yields batches of size 'batch_size' from the given list of sentences"""
    num_sents = len(sentences)
    for idx in range(0, num_sents, batch_size):
        yield sentences[idx : idx + batch_size]


def evaluate(
    model_name: str,
    preprocessed_dataset: str,
    src_lang: str,
    dataset_split: Optional[str] = "test",
    batch_size: int = 8,
    num_beams: int = 5,
    max_length: int = 200,
):
    model, tok_wrapper, logitsprocessor = get_resources(model_name)
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "logits_processor": LogitsProcessorList([logitsprocessor]),
    }

    test_dataset = DatasetDict.load_from_disk(preprocessed_dataset)

    if dataset_split:
        test_dataset = test_dataset[dataset_split]

    sentid = 0
    n_invalid = 0
    references = []
    predictions = []
    for batch in tqdm(batchify(test_dataset), unit="batch", total=max(1, len(test_dataset) // batch_size)):
        for sentence, ref_penman, pred_linearized in zip(
            batch["sentence"],
            batch["penmanstr"],
            batch_translate(
                texts=batch["sentence"], src_lang=src_lang, model=model, tok_wrapper=tok_wrapper, **gen_kwargs
            ),
        ):
            references.append(ref_penman)
            references.append(pred_linearized)
            # TODO: change pred_linearized

    graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
    ilp = solvers.ILP()
    smatch_metric = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)

    score, optimization_status = smatch_metric.score_corpus(references, predictions)
    score = score["main"]
    score = {
        "smatch_precision": score["Precision"],
        "smatch_recall": score["Recall"],
        "smatch_fscore": score["F1"],
    }

    print(f"Evaluation scores of {model_name} on {preprocessed_dataset}")
    print("Smatch: ", f"p={score[0]:.4f},r={score[1]:.4f},f={score[2]:.4f}")
    print("No. invalid structures: ", n_invalid, f"({(n_invalid*100/sentid):.2f}%)")


@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "the directory containing the adapters"})
    preprocessed_dataset: str = field(metadata={"help": "where to save the output"})
    src_lang: str = field(metadata={"help": "which source language to use. This can be a language code"})
    dataset_split: Optional[str] = field(default="test", metadata={"help": "which split of the dataset to use"})
    batch_size: Optional[int] = field(
        default=8, metadata={"help": "batch size (lower this if you get out-of-memory errors)"}
    )
    num_beams: Optional[int] = field(default=5, metadata={"help": "number of beams for generation"})
    max_length: Optional[int] = field(default=200, metadata={"help": "max. length to generate"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    evaluate(
        model_name=script_args.model_name,
        preprocessed_dataset=script_args.preprocessed_dataset,
        src_lang=script_args.src_lang,
        dataset_split=script_args.dataset_split,
        batch_size=script_args.batch_size,
        num_beams=script_args.num_beams,
        max_length=script_args.max_length,
    )


if __name__ == "__main__":
    main()
