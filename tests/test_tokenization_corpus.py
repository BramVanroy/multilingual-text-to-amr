from collections import Counter
from itertools import product
from pathlib import Path
from typing import List
from smatchpp import Smatchpp, preprocess, solvers
from tqdm import tqdm
import penman

from multi_amr.data.linearization import dfs_linearize, remove_wiki_from_graph
from multi_amr.data.postprocessing_str import postprocess_str_after_linearization
from multi_amr.data.tokenization import AMRTokenizerWrapper

USE_FAST = (True, False)
TOKENIZER_NAMES = (
    "bigscience/bloomz-560m",
    "facebook/mbart-large-cc25",
    "facebook/bart-large",
    "facebook/mbart-large-50-many-to-one-mmt",
    "google/mt5-base",
    "t5-base",
    "facebook/nllb-200-3.3B",
    "google/flan-t5-base",
)
REMOVE_WIKI = (True, False)

# USE_FAST = (True,)
# TOKENIZER_NAMES = (
#     "facebook/bart-large",
# )
# REMOVE_WIKI = (True,)


graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
ilp = solvers.ILP()
smatch_metric = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)


def calculate_corpus_smatch(references: List[str], predictions: List[str]):
    score, optimization_status = smatch_metric.score_corpus(references, predictions)
    score = score["main"]
    return {
        "smatch_precision": score["Precision"]["result"],
        "smatch_recall": score["Recall"]["result"],
        "smatch_fscore": score["F1"]["result"],
    }


def calculate_pair_smatch(reference: str, prediction: str):
    score = smatch_metric.score_pair(reference, prediction)
    score = score["main"]
    return {
        "smatch_precision": score["Precision"],
        "smatch_recall": score["Recall"],
        "smatch_fscore": score["F1"],
    }


PENMAN_TEST_EXAMPLE = """(b / buy-01
   :ARG0 (c / country
            :wiki +
            :name (n / name
                     :op1 "India"))
   :ARG1 (s / ship
            :wiki +
            :name (n2 / name
                      :op1 "Trenton"
                      :op2 "(LPD-14)")))
"""



def main_sp(indir: str, single: bool = False):
    TEST_SINGLE_EXAMPLE = single
    pdin = Path(indir)
    runs_stats = {}

    for use_fast, tok_name, rm_wiki in product(USE_FAST, TOKENIZER_NAMES, REMOVE_WIKI):
        if tok_name == "bigscience/bloomz-560m" and not use_fast:
            # BLOOM does not have a slow tokenizer
            continue
        tok_wrapper = AMRTokenizerWrapper.from_pretrained(tok_name, use_fast=use_fast, legacy=True)

        num_not_perfect_smatch = 0
        all_refs_penman = []
        all_linearizeds = []
        status_counter = Counter()
        for pfin in tqdm(list(pdin.rglob("*.txt")),
                         unit="file",
                         desc=f"(read) Remove wiki? {rm_wiki} Tok? {tok_name} Fast? {use_fast}"):
            with pfin.open(encoding="utf-8") as fhin:
                for graph in penman.iterdecode(fhin):
                    if TEST_SINGLE_EXAMPLE:
                        graph = penman.decode(PENMAN_TEST_EXAMPLE)
                    graph.metadata = []

                    if rm_wiki:
                        graph = remove_wiki_from_graph(graph)

                    all_refs_penman.append(penman.encode(graph).replace("–", "-"))
                    linearized_nodes = dfs_linearize(graph)
                    linearized = " ".join(linearized_nodes)
                    linearized = postprocess_str_after_linearization(linearized, verbose=TEST_SINGLE_EXAMPLE)
                    all_linearizeds.append(linearized)
                    if TEST_SINGLE_EXAMPLE:
                        break
            if TEST_SINGLE_EXAMPLE:
                break

        if not all_refs_penman:
            continue

        all_preds_penman = []
        for linearized, ref_penman in tqdm(zip(all_linearizeds, all_refs_penman),
                                           total=len(all_linearizeds),
                                           desc=f"(compare) Remove wiki? {rm_wiki} Tok? {tok_name} Fast? {use_fast}"):
            encoded = tok_wrapper(linearized)
            try:
                pred_graph, status, _ = tok_wrapper.decode_amr_ids(encoded.input_ids, verbose=TEST_SINGLE_EXAMPLE)
                pred_penman = penman.encode(pred_graph)
            except Exception as exc:
                print(f"Error decoding")
                print("linearized input:", linearized)
                print("input ids:", encoded.input_ids)
                print("converted input ids:", tok_wrapper.tokenizer.convert_ids_to_tokens(encoded.input_ids))
                print("ref:", ref_penman)
                print()
                raise exc

            all_preds_penman.append(pred_penman)
            status_counter[status] += 1

            # This is slow, but sometimes graphs can be an exact match even if they look slightly different
            pair_score = calculate_pair_smatch(ref_penman, pred_penman)
            if pair_score["smatch_fscore"] < 100.:
                print(f"Not an exact graph match")
                print("linearized input:", linearized)
                print("ref:", ref_penman)
                print("pred:", pred_penman)
                print()
                num_not_perfect_smatch += 1

        assert len(all_preds_penman) == len(all_refs_penman) == len(all_linearizeds)

        # Calculate corpus-level smatch
        score = calculate_corpus_smatch(all_refs_penman, all_preds_penman)
        print("SCORE", score)
        status_stats = "; ".join([f"{status}: {num:,}" for status, num in status_counter.items()])
        runs_stats[(rm_wiki, tok_name, use_fast)] = {
            "status_stats": status_stats,
            "num_not_perfect_match": num_not_perfect_smatch,
            "percent_not_perfect_match": f"{(100*num_not_perfect_smatch / status_counter.total()):.2f}%",
            "smatch": {k: f"{v:.4f}" for k, v in score.items()}
        }

    for params, stats in runs_stats.items():
        remove_wiki, tokenizer_name, use_fast = params
        print(f"Remove wiki? {remove_wiki} Tok? {tokenizer_name} Fast? {use_fast}")
        print(stats)
        print()



if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Brute-force testing of AMR tokenization by tokenizing a linearized "
                                                  "tree, tokenizing it, decoding it, and reconstructing the tree."
                                                  " Then checking if the original and reconstructed trees are equal."
                                                  " This script is a naive, brute-force way of testing this. All .txt"
                                                  " files in a given directory will be (recursively) tested.")
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. Will try to read"
                                       " this as a HF dataset. If not possible, all .txt files will be tested.")
    cparser.add_argument("--single", action="store_true", help="Only process the single graph at the top of code (for debugging).")

    cargs = cparser.parse_args()
    main_sp(cargs.indir, cargs.single)
