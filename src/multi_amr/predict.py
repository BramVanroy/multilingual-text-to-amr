import json
from dataclasses import dataclass, field
from glob import glob
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple

import penman
import torch
from multi_amr.data.postprocessing_graph import ParsedStatus
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from penman.models.noop import model as noop_model
from smatchpp import Smatchpp, preprocess, solvers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, HfArgumentParser


@dataclass
class PredictArguments:
    checkpoint: str = field(metadata={"help": "the directory containing the adapters"})
    datasets: List[str] = field(
        metadata={
            "help": "one or more glob patterns (e.g. ending with `test/*.txt`). If `predict_from_plaintext=True`, will read these files line-by-line and simply generate predictions"
        }
    )
    predict_from_plaintext: bool = field(
        default=False,
        metadata={
            "help": "whether to read the input files as plain text, and for every line generate a prediction. No scoring will be done."
        },
    )

    num_beams: int = field(default=5, metadata={"help": "how many beams to use during generation"})
    batch_size: int = field(
        default=4, metadata={"help": "batch size. Lower this if you are getting out of memory errors"}
    )
    max_new_tokens: int = field(default=1024, metadata={"help": "Max number of new tokens to generate"})
    no_cuda: bool = field(default=False, metadata={"help": "Whether to disable CUDA."})
    output_file_predictions: Optional[str] = field(
        default=None, metadata={"help": "If given, will write the predictions to this file."}
    )
    output_path_score: Optional[str] = field(
        default=None,
        metadata={
            "help": "If given, will write the smatch score predictions to this directory in 'test_results.json'."
        },
    )


def calculate_smatch(ref_penmans: List[str], pred_penmans: List[str]):
    graph_standardizer = preprocess.AMRStandardizer()
    solver = solvers.ILP()
    smatch_metric = Smatchpp(alignmentsolver=solver, graph_standardizer=graph_standardizer)

    score, _ = smatch_metric.score_corpus(ref_penmans, pred_penmans)
    return score


def batchify(texts, batch_size: int = 4):
    for idx in range(0, len(texts), batch_size):
        yield texts[idx : idx + batch_size]


def predict(
    model,
    tok_wrapper: AMRTokenizerWrapper,
    texts: List[str],
    *,
    batch_size: int = 4,
    num_beams: int = 5,
    max_new_tokens: int = 1024,
) -> List[Tuple[penman.Graph, ParsedStatus]]:
    # TODO add forced_bos_token_id for multilingual models so that the first token is set to the AMR token?
    predictions = []
    with torch.no_grad():
        for batch_texts in tqdm(
            batchify(texts, batch_size=batch_size), unit="batch", total=ceil(len(texts) / batch_size)
        ):
            inputs = tok_wrapper(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            output = model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=num_beams,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                decoder_start_token_id=tok_wrapper.amr_token_id,
            )
            for idx in range(0, output.sequences.size(0), num_beams):
                sample_sequences = output.sequences[idx : idx + num_beams]
                sample_seq_scores = output.sequences_scores[idx : idx + num_beams].tolist()
                decoded = tok_wrapper.batch_decode_amr_ids(sample_sequences)
                # OK (0) first, then FIXED (1), then BACKOFF (2)
                # Highest scored prediction first (so sort from high to low thanks to "-")
                sorted_by_status_and_score = sorted(
                    zip(decoded["graph"], decoded["status"], sample_seq_scores),
                    key=lambda sample: (sample[1].value, -sample[2]),
                )

                # Only return best item, do not include scores
                predictions.append(sorted_by_status_and_score[0][:2])

    return predictions


def read_data(datasets: List[str], predict_from_plaintext: bool = False) -> Tuple[List[str], List[penman.Graph]]:
    texts = []
    ref_graphs = []
    for fin_pattern in datasets:
        for fstr in glob(fin_pattern):
            lines = Path(fstr).read_text(encoding="utf-8").splitlines()
            if predict_from_plaintext:
                texts.extend(lines)
            else:
                for graph in penman.iterdecode(lines, model=noop_model):
                    ref_graphs.append(graph)
                    try:
                        texts.append(graph.metadata["snt"])
                    except KeyError as exc:
                        raise KeyError(
                            "To evaluate given graphs, the graphs must contain a sentence as metadata ('snt')."
                        ) from exc

    return texts, ref_graphs


def main():
    parser = HfArgumentParser(PredictArguments)
    args = parser.parse_args_into_dataclasses()[0]
    tok_wrapper = AMRTokenizerWrapper.from_pretrained(args.checkpoint)

    if tok_wrapper.tokenizer_type in (
        TokenizerType.MBART,
        TokenizerType.NLLB,
        TokenizerType.BART,
        TokenizerType.T5,
    ):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.eval()

    if not args.no_cuda:
        model.to("cuda")

    texts, ref_graphs = read_data(args.datasets, args.predict_from_plaintext)

    predictions = predict(model, tok_wrapper, texts, batch_size=args.batch_size, num_beams=args.num_beams)
    pred_graphs = list(zip(*predictions))[0]

    ref_penmans = []
    for graph in ref_graphs:
        graph.metadata = {}
        ref_penmans.append(penman.encode(graph))

    pred_penmans = []
    for graph in pred_graphs:
        graph.metadata = {}
        pred_penmans.append(penman.encode(graph))

    score = calculate_smatch(ref_penmans, pred_penmans)
    try:
        score = score["main"]
    except KeyError:
        pass
    print(score)
    if args.output_path_score:
        Path(args.output_path_score).joinpath("test_results.json").write_text(json.dumps(score), encoding="utf-8")


if __name__ == "__main__":
    main()
