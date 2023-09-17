from typing import List
from collections import Counter

from smatchpp import preprocess, solvers, eval_statistics

from smatchpp import Smatchpp, util
from multi_amr.data.postprocessing_graph import BACKOFF
import penman
from pathlib import Path
import pandas as pd


class BackOffSmatchpp(Smatchpp):
    def process_corpus(self, amrs, amrs2):
        status = []
        match_dict = {}
        # Track the indices where we had to backoff
        back_offed_idxs = []
        for i, a in enumerate(amrs):
            try:
                match, tmpstatus, _ = self.process_pair(a, amrs2[i])
            except Exception as exc:
                back_offed_idxs.append(i)
                match, tmpstatus, _ = self.process_pair(a, penman.encode(BACKOFF))
            status.append(tmpstatus)
            util.append_dict(match_dict, match)
        return match_dict, status, back_offed_idxs

    def score_corpus(self, amrs, amrs2):
        match_dict, status, back_offed_idxs = self.process_corpus(amrs, amrs2)

        # pairwise statistic
        if self.printer.score_type is None:
            final_result = []
            for i in range(len(amrs)):
                match_dict_tmp = {k: [match_dict[k][i]] for k in match_dict.keys()}
                result = self.printer.get_final_result(match_dict_tmp)
                final_result.append(result)

        # aggregate statistic (micro, macro...)
        else:
            final_result = self.printer.get_final_result(match_dict)
        return final_result, status, back_offed_idxs


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


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        "Generate AMR from text with OpenAI's API.\nIf you get a RateLimitError concerning using more tokens per minute"
        " than is accepted, you can try lowering --max_parallel_requests to a smaller number.\n"
        " To use this script, you need access to the OpenAI API. Make sure your API key is set as an environment"
        " variable OPENAI_API_KEY (or use --api_key). Note: THIS WILL INCUR COSTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument("fpred", help="Input TSV file with columns 'uid' and 'penman_str'")
    cparser.add_argument("dref", help="Directory with references (amr/test files)")
    cargs = cparser.parse_args()
    predictions, references, statuses = read_data(cargs.fpred, cargs.dref)
    score, back_offed_idxs = score_corpus(predictions, references)

    print("SCORE", score)
    for back_off_idx in back_offed_idxs:
        statuses[back_off_idx] = "backoff"

    status_counter = Counter(statuses)
    print("STATUS COUNTER", status_counter)
    print("NO. NEW BACKOFFS", len(back_offed_idxs))


if __name__ == '__main__':
    main()
