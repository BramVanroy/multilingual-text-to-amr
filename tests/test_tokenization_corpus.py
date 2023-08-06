from collections import Counter
from itertools import product
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Optional, Dict

from tqdm import tqdm
import penman

from multi_amr.data.postprocessing_str import postprocess_str_after_linearization
from multi_amr.data.prepare_dataset import remove_wiki_from_graph, dfs_linearize
from multi_amr.data.tokenization import AMRTokenizerWrapper
from multi_amr.utils.calculate_smatch import calculate_smatch

USE_FAST = (True, False)
TOKENIZER_NAMES = (
    "bigscience/bloomz-560m",
    "facebook/mbart-large-cc25",
    "google/mt5-base",
    "facebook/nllb-200-3.3B"
)
REMOVE_WIKI = (True, False)


def main_sp(indir: str, start_from: Optional[int] = None):
    # Not the same as main_mp/workers: here we can keep track of sequential graph_idx to skip forward if needed
    # Useful for debugging
    pdin = Path(indir)
    graph_idx = 0
    runs_stats = {}

    for use_fast in USE_FAST:
        for tokenizer_name in TOKENIZER_NAMES:
            if tokenizer_name == "bigscience/bloomz-560m" and not use_fast:
                # BLOOM does not have a slow tokenizer
                continue
            tok_wrapper = AMRTokenizerWrapper.from_pretrained(tokenizer_name, use_fast=use_fast, legacy=True)

            for remove_wiki in REMOVE_WIKI:
                status_counter = Counter()
                num_not_perfect_smatch = 0
                for pfin in tqdm(list(pdin.rglob("*.txt")),
                                 unit="file",
                                 desc=f"Remove wiki? {remove_wiki} Tok? {tokenizer_name} Fast? {use_fast}"):
                    with pfin.open(encoding="utf-8") as fhin:
                        linearizeds = []
                        refs_penman = []
                        for graph in penman.iterdecode(fhin):
                            graph_idx += 1
                            if start_from is not None and start_from > 0 and graph_idx < start_from:
                                continue

                            graph.metadata = []
                            if remove_wiki:
                                graph = remove_wiki_from_graph(graph)

                            linearized = " ".join(dfs_linearize(graph))
                            linearized = postprocess_str_after_linearization(linearized)
                            linearizeds.append(linearized)
                            refs_penman.append(penman.encode(graph))

                        encoded = tok_wrapper(linearizeds)
                        output = tok_wrapper.batch_decode_amr_ids(encoded.input_ids, verbose=True)

                        for ref_penmanstr, pred_penmanstr, status in zip(refs_penman, output["penman"], output["status"]):
                            status_counter[status.name] += 1
                            score = calculate_smatch([ref_penmanstr], [pred_penmanstr])
                            if score["smatch_fscore"] != 1:
                                print("Not a good match!!")
                                print("ref:", ref_penmanstr)
                                print("pred:", pred_penmanstr)
                                print()
                                num_not_perfect_smatch += 1

                status_stats = "; ".join([f"{status}: {num:,}" for status, num in status_counter.items()])
                runs_stats[(remove_wiki, tokenizer_name, use_fast)] = {
                    "status_stats": status_stats,
                    "num_not_perfect_smatch": num_not_perfect_smatch,
                    "percent_not_perfect_smatch": 100*num_not_perfect_smatch / status_counter.total(),
                }
                print((remove_wiki, tokenizer_name, use_fast))
                print(runs_stats[(remove_wiki, tokenizer_name, use_fast)])

    for params, stats in runs_stats.items():
        remove_wiki, tokenizer_name, use_fast = params
        print(f"Remove wiki? {remove_wiki} Tok? {tokenizer_name} Fast? {use_fast}")
        print(stats)
        print()


def worker(pdin: Path, runs_stats: Dict, use_fast: bool, tokenizer_name: str, remove_wiki: bool):
    tok_wrapper = AMRTokenizerWrapper.from_pretrained(tokenizer_name, use_fast=use_fast, legacy=True)
    status_counter = Counter()
    num_not_perfect_smatch = 0
    for pfin in tqdm(list(pdin.rglob("*.txt")),
                     unit="file",
                     desc=f"Remove wiki? {remove_wiki} Tok? {tokenizer_name} Fast? {use_fast}"):
        with pfin.open(encoding="utf-8") as fhin:
            linearizeds = []
            refs_penman = []
            for graph in penman.iterdecode(fhin):
                graph.metadata = []
                if remove_wiki:
                    graph = remove_wiki_from_graph(graph)

                linearized = " ".join(dfs_linearize(graph))
                linearized = postprocess_str_after_linearization(linearized)
                linearizeds.append(linearized)
                refs_penman.append(penman.encode(graph))

            encoded = tok_wrapper(linearizeds)
            output = tok_wrapper.batch_decode_amr_ids(encoded.input_ids, verbose=True)

            for ref_penmanstr, pred_penmanstr, status in zip(refs_penman, output["penman"], output["status"]):
                status_counter[status.name] += 1
                score = calculate_smatch([ref_penmanstr], [pred_penmanstr])
                if score["smatch_fscore"] != 1:
                    print("Not a good match!!")
                    print("ref:", ref_penmanstr)
                    print()
                    num_not_perfect_smatch += 1

    status_stats = "; ".join([f"{status}: {num:,}" for status, num in status_counter.items()])
    runs_stats[(remove_wiki, tokenizer_name, use_fast)] = {
                "status_stats": status_stats,
                "num_not_perfect_smatch": num_not_perfect_smatch,
                "percent_not_perfect_smatch": 100*num_not_perfect_smatch / status_counter.total(),
            }

    return runs_stats


def main_mp(indir: str, num_workers: int):
    pdin = Path(indir)

    with Manager() as manager:
        runs_stats = manager.dict()

        with Pool(num_workers) as pool:
            pool.starmap(
                worker,
                [
                    (pdin, runs_stats, use_fast, tok_name, rm_wiki)
                    for use_fast, tok_name, rm_wiki in product(USE_FAST, TOKENIZER_NAMES, REMOVE_WIKI)
                ]
            )

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
        main_mp(cargs.indir, cargs.num_workers)
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
