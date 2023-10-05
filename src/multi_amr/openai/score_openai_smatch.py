from collections import Counter
from os import PathLike
from pathlib import Path
from typing import List

import pandas as pd
import penman
from smatchpp import eval_statistics, preprocess, solvers

from multi_amr.evaluate.backoff_smatch import BackOffSmatchpp


def read_data(fpred: str, dref: str):
    df = pd.read_csv(fpred, sep="\t", encoding="utf-8")
    for pfin in Path(dref).rglob("*.txt"):
        with pfin.open(encoding="utf-8") as fhin:
            for graph in penman.iterdecode(fhin):
                sentid = graph.metadata["id"]
                fname = pfin.name
                uid = f"{fname}__{sentid}"
                graph.metadata = {}
                df.loc[df["uid"] == uid, "reference"] = penman.encode(graph)

    references = df["reference"].tolist()
    predictions = df["penman_str"].tolist()
    statuses = df["status"].tolist()
    assert len(references) == len(predictions)

    print(references[0])
    print(predictions[0])
    return predictions, references, statuses


def score_corpus(predictions: List[str], references: List[str]):
    graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
    ilp = solvers.ILP()
    printer = eval_statistics.ResultPrinter(score_type="micro", do_bootstrap=True, output_format="json")
    smatch_metric = BackOffSmatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer, printer=printer)

    score, optimization_status, back_offed_idxs = smatch_metric.score_corpus(references, predictions)
    score = score["main"]
    score = {
        "smatch_precision": score["Precision"],
        "smatch_recall": score["Recall"],
        "smatch_fscore": score["F1"],
    }
    return score, back_offed_idxs


def score_file(fpred: str | PathLike):
    df = pd.read_csv(fpred, encoding="utf-8", sep="\t")
    predictions = df["penman_str"].tolist()
    statuses = df["status"].tolist()
    references = df["reference_penman_str"].tolist()

    score, back_offed_idxs = score_corpus(predictions, references)

    print("SCORE", score)
    for back_off_idx in back_offed_idxs:
        statuses[back_off_idx] = "backoff"

    status_counter = Counter(statuses)
    print("STATUS COUNTER", status_counter)
    print("NO. NEW BACKOFFS", len(back_offed_idxs))


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        "Score the AMRs generated with OpenAI's API. Will use BACKOFF graph in case of issues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "fpred", help="Input TSV file with columns 'reference_penman_str' and 'penman_str' and 'status'"
    )
    cargs = cparser.parse_args()
    score_file(cargs.fpred)


if __name__ == "__main__":
    main()
