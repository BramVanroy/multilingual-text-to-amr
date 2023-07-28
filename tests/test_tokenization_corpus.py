from pathlib import Path
from typing import Optional

from tqdm import tqdm
import penman
from ftfy import fix_text

from amr import AMR

from multi_amr.data.linearization import penmanstr2linearized, linearized2penmanstr, do_remove_wiki
from multi_amr.data.tokenization import AMRMBartTokenizer


def main(indir: str, start_from: Optional[int] = None):
    pdin = Path(indir)
    tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="amr_XX")

    tree_idx = 0
    for remove_wiki in (True, False):
        for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file", desc=f"Remove wiki? {remove_wiki}", disable=True):
            with pfin.open(encoding="utf-8") as fhin:
                for tree in penman.iterparse(fhin):
                    tree_idx += 1
                    if start_from is not None and start_from > 0 and tree_idx < start_from:
                        continue
                    tree.reset_variables()
                    orig_text = fix_text(tree.metadata["snt"])
                    # NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
                    # (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                    penman_str = fix_text(penman.format(tree))

                    if remove_wiki:
                        penman_str = do_remove_wiki(penman_str)

                    original_tree = penman.parse(penman_str)

                    encoded = tokenizer.encode_penmanstrs(penman_str, remove_wiki=remove_wiki)
                    decoded = tokenizer.decode_and_fix(encoded.input_ids)[0]

                    try:
                        delinearized_penman_str = linearized2penmanstr(decoded)
                        delinearized_tree = penman.parse(delinearized_penman_str)
                    except penman.exceptions.DecodeError as exc:
                        print("META", tree.metadata)
                        print("ORIG TREE", original_tree)
                        print("DECODED", decoded)
                        print("FILE", str(pfin))
                        print("WIKI", remove_wiki)
                        print("TREE ID", tree_idx)
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
                        print("DELINEARIZED PENMAN", delinearized_penman_str)
                        print("DELINEARIZED TREE", delinearized_tree)
                        print("FILE", str(pfin))
                        print("WIKI", remove_wiki)
                        print("TREE ID", tree_idx)
                        raise exc

                    if original_tree != delinearized_tree:
                        print(tree.metadata)
                        print("PENMAN", penman_str)
                        print("LINEARIZED", penmanstr2linearized(penman_str))
                        print("ENCODED", encoded["input_ids"])
                        print("DECODED", tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True))
                        print("DECODED_FIXED", decoded)
                        print("ORIG TREE", original_tree)
                        print("DELINEARIZED TREE", delinearized_tree)
                        print("FILE", str(pfin))
                        print("WIKI", remove_wiki)
                        print("TREE ID", tree_idx)
                        raise ValueError("Tree mismatch between original tree and delinearized tree")

                    # Check text tokenization
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
