from collections import Counter
from itertools import product
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Optional, Dict, List
from smatchpp import Smatchpp, preprocess, solvers
from ftfy import fix_text
from tqdm import tqdm
import penman

from multi_amr.data.postprocessing_graph import reorder_graph_triples
from multi_amr.data.postprocessing_str import postprocess_str_after_linearization
from multi_amr.data.linearization import remove_wiki_from_graph, dfs_linearize
from multi_amr.data.tokenization import AMRTokenizerWrapper

USE_FAST = (True, False)
TOKENIZER_NAMES = (
    "bigscience/bloomz-560m",
    "facebook/mbart-large-cc25",
    "google/mt5-base",
    "facebook/nllb-200-3.3B"
)
REMOVE_WIKI = (True, False)
# USE_FAST = (True,)
# TOKENIZER_NAMES = (
#     "facebook/mbart-large-cc25",
# )
# REMOVE_WIKI = (False,)


graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
ilp = solvers.ILP()
smatch_metric = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)


def calculate_smatch(references: List[str], predictions: List[str]):
    score, optimization_status = smatch_metric.score_corpus(references, predictions)
    score = score["main"]
    return {
        "smatch_precision": score["Precision"]["result"],
        "smatch_recall": score["Recall"]["result"],
        "smatch_fscore": score["F1"]["result"],
    }


def main_sp(indir: str, start_from: Optional[int] = None):
    # Not the same as main_mp/workers: here we can keep track of sequential graph_idx to skip forward if needed
    # Useful for debugging
    pdin = Path(indir)
    graph_idx = 0
    runs_stats = {}

    for use_fast, tok_name, rm_wiki in product(USE_FAST, TOKENIZER_NAMES, REMOVE_WIKI):
        if tok_name == "bigscience/bloomz-560m" and not use_fast:
            # BLOOM does not have a slow tokenizer
            continue
        tok_wrapper = AMRTokenizerWrapper.from_pretrained(tok_name, use_fast=use_fast, legacy=True)

        num_not_perfect_smatch = 0
        all_refs_penman = []
        all_linearizeds = []
        all_preds_penman = []
        status_counter = Counter()
        for pfin in tqdm(list(pdin.rglob("*.txt")),
                         unit="file",
                         desc=f"Remove wiki? {rm_wiki} Tok? {tok_name} Fast? {use_fast}"):
            with pfin.open(encoding="utf-8") as fhin:
                for graph in penman.iterdecode(fhin):
                    graph_idx += 1
                    if start_from is not None and start_from > 0 and graph_idx < start_from:
                        continue

                    graph.metadata = []
                    if rm_wiki:
                        graph = remove_wiki_from_graph(graph)

                    # Mostly needed to ensure depth-first order for exact matching of the graph
                    # In terms of smatch score this should not matter
                    graph = reorder_graph_triples(graph)
                    # NLLB does not support en-dashes -> normalize
                    cleaned_punct_penman = fix_text(penman.encode(graph).replace("–", "-"))
                    graph = penman.decode(cleaned_punct_penman)

                    linearized = " ".join(dfs_linearize(graph))
                    linearized = postprocess_str_after_linearization(linearized)
                    all_linearizeds.append(linearized)

                    tree = penman.configure(graph)
                    tree.reset_variables()
                    ref_penman = penman.format(tree)
                    all_refs_penman.append(ref_penman)

        for linearized, ref_penman in zip(all_linearizeds, all_refs_penman):
            encoded = tok_wrapper(linearized)
            pred_penman, status = tok_wrapper.decode_amr_ids(encoded.input_ids, reset_variables=True)

            all_preds_penman.append(pred_penman)
            status_counter[status] += 1

            ref_graph = penman.decode(ref_penman)
            pred_graph = penman.decode(pred_penman)
            if ref_graph != pred_graph:
                print(f"Not an exact graph match")
                print("linearized input:", linearized)
                print("ref:", ref_penman)
                print("pred:", pred_penman)
                print("idx:", graph_idx)
                print()
                num_not_perfect_smatch += 1

        assert len(all_preds_penman) == len(all_refs_penman) == len(all_linearizeds)

        # Calculate corpus-level smatch
        score = calculate_smatch(all_refs_penman, all_preds_penman)
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
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. All .txt files"
                                       " will be tested.")
    cparser.add_argument("--start", type=int, help="Start processing from this tree index. Useful if you know exactly"
                                                   " where something went wrong. Won't work in multiprocessing")
    cparser.add_argument("--num_workers", type=int, default=1,
                         help="If > 1, will launch parallel processes to run tests")
    cargs = cparser.parse_args()
    if cargs.num_workers > 1:
        raise NotImplementedError("Multi-processing not implemented")
    else:
        main_sp(cargs.indir, cargs.start)

"""
RESULTS
=======
Remove wiki? True Tok? bigscience/bloomz-560m Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9996, 'smatch_recall': 0.9996, 'smatch_fscore': 0.9996, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? bigscience/bloomz-560m Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9984, 'smatch_recall': 0.9984, 'smatch_fscore': 0.9984, 'ratio_invalid_amrs': 0.0}}

Remove wiki? True Tok? facebook/mbart-large-cc25 Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9992, 'smatch_recall': 0.9989, 'smatch_fscore': 0.9991, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? facebook/mbart-large-cc25 Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9982, 'smatch_recall': 0.9982, 'smatch_fscore': 0.9982, 'ratio_invalid_amrs': 0.0}}

Remove wiki? True Tok? google/mt5-base Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9991, 'smatch_recall': 0.9988, 'smatch_fscore': 0.9990, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? google/mt5-base Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9982, 'smatch_recall': 0.9981, 'smatch_fscore': 0.9981, 'ratio_invalid_amrs': 0.0}}

Remove wiki? True Tok? facebook/nllb-200-3.3B Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9992, 'smatch_recall': 0.9989, 'smatch_fscore': 0.9991, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? facebook/nllb-200-3.3B Fast? True
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9982, 'smatch_recall': 0.9982, 'smatch_fscore': 0.9982, 'ratio_invalid_amrs': 0.0}}

Remove wiki? True Tok? facebook/mbart-large-cc25 Fast? False
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9991, 'smatch_recall': 0.9988, 'smatch_fscore': 0.9990, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? facebook/mbart-large-cc25 Fast? False
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9982, 'smatch_recall': 0.9982, 'smatch_fscore': 0.9982, 'ratio_invalid_amrs': 0.0}}

Remove wiki? True Tok? google/mt5-base Fast? False
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9976, 'smatch_recall': 0.9973, 'smatch_fscore': 0.9975, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? google/mt5-base Fast? False
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9968, 'smatch_recall': 0.9967, 'smatch_fscore': 0.9968, 'ratio_invalid_amrs': 0.0}}

Remove wiki? True Tok? facebook/nllb-200-3.3B Fast? False
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9991, 'smatch_recall': 0.9988, 'smatch_fscore': 0.9990, 'ratio_invalid_amrs': 0.0}}

Remove wiki? False Tok? facebook/nllb-200-3.3B Fast? False
{'status_stats': 'OK: 1,722', 'smatch': {'smatch_precision': 0.9981, 'smatch_recall': 0.9980, 'smatch_fscore': 0.9980, 'ratio_invalid_amrs': 0.0}}"""
