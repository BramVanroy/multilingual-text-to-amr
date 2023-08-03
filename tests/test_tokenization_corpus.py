from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
import penman
from ftfy import fix_text

from amr import AMR

from multi_amr.data.linearization import penmanstr2linearized, linearized2penmanstr, do_remove_wiki
from multi_amr.data.tokenization import AMRTokenizerWrapper


def main(indir: str, start_from: Optional[int] = None):
    pdin = Path(indir)
    tree_idx = 0
    # Replace en-dash by regular dash
    normalize_punct = lambda s: s.replace("–", "-")
    for use_fast in (True, False):
        for tokenizer_name in (
                "bigscience/bloomz-560m",
                "facebook/mbart-large-cc25",
                "google/mt5-base",
                "facebook/nllb-200-3.3B"
        ):
            if tokenizer_name == "bigscience/bloomz-560m" and not use_fast:
                # BLOOM does not have a slow tokenizer
                continue
            tok_wrapper = AMRTokenizerWrapper.from_pretrained(tokenizer_name, use_fast=use_fast, legacy=True)

            for remove_wiki in (True, False):
                for pfin in tqdm(list(pdin.rglob("*.txt")),
                                 unit="file",
                                 desc=f"Remove wiki? {remove_wiki} Tok? {tokenizer_name} Fast? {use_fast}"):
                    with pfin.open(encoding="utf-8") as fhin:
                        for tree in penman.iterparse(fhin):
                            tree_idx += 1
                            if start_from is not None and start_from > 0 and tree_idx < start_from:
                                continue
                            tree.reset_variables()
                            orig_text = normalize_punct(fix_text(tree.metadata["snt"]))
                            # NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
                            # (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                            penman_str = normalize_punct(fix_text(penman.format(tree)))

                            if remove_wiki:
                                penman_str = do_remove_wiki(penman_str)

                            original_tree = penman.parse(penman_str)

                            linearized = penmanstr2linearized(penman_str, remove_wiki=remove_wiki)
                            encoded = tok_wrapper([linearized], padding=False, truncation=False, return_tensors="pt")
                            input_ids = encoded["input_ids"]
                            # Replace all the language IDs with the amr_token_id
                            if tok_wrapper.lang_idxs is not None:
                                input_ids[torch.isin(input_ids, tok_wrapper.lang_idxs)] = tok_wrapper.amr_token_idx

                            decoded = tok_wrapper.decode_and_fix_amr(input_ids)[0]

                            try:
                                delinearized_penman_str = linearized2penmanstr(decoded)
                                delinearized_tree = penman.parse(delinearized_penman_str)
                            except penman.exceptions.DecodeError as exc:
                                print("META", tree.metadata)
                                print("ORIG TREE", original_tree)
                                print("LINEARIZED", linearized)
                                print("DECODED", decoded)
                                print("FILE", str(pfin))
                                print("WIKI", remove_wiki)
                                print("TREE ID", tree_idx)
                                print("TOKENIZER", tokenizer_name)
                                print("USE FAST", use_fast)
                                raise exc

                            try:
                                # Although we use penman to parse AMR, `smatch` uses its own AMR parser
                                # so we also verify that our linearization/delinearization works with that
                                amr_parsed = AMR.parse_AMR_line(delinearized_penman_str)

                                if amr_parsed is None:
                                    raise Exception("Error parsing AMR with amr.AMR library")
                            except Exception as exc:
                                print("META", tree.metadata)
                                print("ORIG TREE", original_tree)
                                print("LINEARIZED", linearized)
                                print("DECODED", decoded)
                                print("DELINEARIZED PENMAN", delinearized_penman_str)
                                print("DELINEARIZED TREE", delinearized_tree)
                                print("FILE", str(pfin))
                                print("WIKI", remove_wiki)
                                print("TREE ID", tree_idx)
                                print("TOKENIZER", tokenizer_name)
                                print("USE FAST", use_fast)
                                raise exc

                            if original_tree != delinearized_tree:
                                print(tree.metadata)
                                print("PENMAN", penman_str)
                                print("LINEARIZED", penmanstr2linearized(penman_str))
                                print("ENCODED", encoded["input_ids"])
                                print("DECODED", tok_wrapper.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True))
                                print("DECODED_FIXED", decoded)
                                print("ORIG TREE", original_tree)
                                print("DELINEARIZED TREE", delinearized_tree)
                                print("FILE", str(pfin))
                                print("WIKI", remove_wiki)
                                print("TREE ID", tree_idx)
                                print("TOKENIZER", tokenizer_name)
                                print("USE FAST", use_fast)
                                raise ValueError("Tree mismatch between original tree and delinearized tree")

                            # # Check text tokenization
                            # input_ids = tokenizer(orig_text)["input_ids"]
                            # detokenized_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                            #
                            # if orig_text != detokenized_text:
                            #     print("ORIGI TEXT", orig_text)
                            #     print("DETOK TEXT", detokenized_text)


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
                                                   " where something went wrong.")
    cargs = cparser.parse_args()
    main(cargs.indir, cargs.start)
