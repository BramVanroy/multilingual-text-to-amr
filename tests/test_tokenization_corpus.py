from collections import Counter
from itertools import product
from pathlib import Path
from typing import List
from smatchpp import Smatchpp, preprocess, solvers
from ftfy import fix_text
from tqdm import tqdm
import penman
from unidecode import unidecode
from penman import Triple

from multi_amr.data.additional_tokens import SPECIAL_ENTITIES_MAP
from multi_amr.data.postprocessing_graph import reorder_graph_triples
from multi_amr.data.postprocessing_str import postprocess_str_after_linearization
from multi_amr.data.linearization import remove_wiki_from_graph, dfs_linearize
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
REPLACE_ENTITIES = (True,)
USE_FAST = (True,)
TOKENIZER_NAMES = (
    "facebook/bart-large",
)
REMOVE_WIKI = (True,)


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


def main_sp(indir: str):
    pdin = Path(indir)
    graph_idx = 0
    runs_stats = {}

    for use_fast, tok_name, rm_wiki, replace_entities in product(USE_FAST, TOKENIZER_NAMES, REMOVE_WIKI, REPLACE_ENTITIES):
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
                    graph.metadata = []

                    if "amr-unintelligible" in penman.encode(graph):
                        # Ignore graphs that contain intelligible utterances
                        continue

                    if rm_wiki:
                        graph = remove_wiki_from_graph(graph)

                    if replace_entities:
                        varmap = {}

                        # Find vars that are a url, email or tel
                        for source, role, target in graph.triples:
                            if role == ":instance":
                                if target in SPECIAL_ENTITIES_MAP:
                                    varmap[source] = target

                        new_triples = []
                        # Find values that refer to url, email or tel vars and replace with special token
                        for source, role, target in graph.triples:
                            # In exceptional cases, the actual URL may be hidden under an op instead of value
                            if source in varmap and role.startswith((":value", ":op")) and target.startswith(
                                    '"') and target.endswith('"'):
                                ent_type = varmap[source]
                                repl = SPECIAL_ENTITIES_MAP[ent_type]
                                new_triples.append(Triple(source, role, repl))
                            else:
                                new_triples.append(Triple(source, role, target))

                        graph = penman.Graph(new_triples, metadata=graph.metadata)

                    # Mostly needed to ensure depth-first order for exact matching of the graph
                    # In terms of smatch score this should not matter
                    graph = reorder_graph_triples(graph)
                    # NLLB does not support en-dashes -> normalize
                    cleaned_punct_penman = fix_text(penman.encode(graph).replace("–", "-"))
                    if not rm_wiki:
                        # `Erdoğan` -> `Erdogan`: useful for models that do not support special characters
                        # This is only needed in wiki entries - in other places they are already normalized
                        # In practice this is NOT a good idea because if we change the wiki entry, the wiki page cannot
                        # be found anymore. We do it here to make sure that our testing works
                        # In reality, we train without wiki anyway so it does not matter...
                        cleaned_punct_penman = unidecode(cleaned_punct_penman)

                    graph = penman.decode(cleaned_punct_penman)

                    linearized = " ".join(dfs_linearize(graph))
                    linearized = postprocess_str_after_linearization(linearized)
                    all_linearizeds.append(linearized)

                    tree = penman.configure(graph)
                    tree.reset_variables()
                    ref_penman = penman.format(tree)
                    all_refs_penman.append(ref_penman)

        if not all_refs_penman:
            continue

        for linearized, ref_penman in zip(all_linearizeds, all_refs_penman):
            encoded = tok_wrapper(linearized)
            try:
                pred_penman, status = tok_wrapper.decode_amr_ids(encoded.input_ids, reset_variables=True)
            except Exception as exc:
                print(f"Error decoding")
                print("linearized input:", linearized)
                print("input ids:", encoded.input_ids)
                print("converted input ids:", tok_wrapper.tokenizer.convert_ids_to_tokens(encoded.input_ids))
                print("ref:", ref_penman)
                print("idx:", graph_idx)
                print()
                raise exc

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
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. Will try to read"
                                       " this as a HF dataset. If not possible, all .txt files will be tested.")

    cargs = cparser.parse_args()
    main_sp(cargs.indir)
