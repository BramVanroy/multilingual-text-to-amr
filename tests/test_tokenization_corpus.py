from pathlib import Path
from tqdm import tqdm
import penman
from ftfy import fix_text

from amr import AMR

from mbart_amr.data.linearization import penmanstr2linearized, linearized2penmanstr, do_remove_wiki
from mbart_amr.data.tokenization import AMRMBartTokenizer


def main(indir: str):
    pdin = Path(indir)
    tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="amr_XX")

    for remove_wiki in (True, False):
        for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file", desc=f"Remove wiki? {remove_wiki}"):
            with pfin.open(encoding="utf-8") as fhin:
                for tree in penman.iterparse(fhin):
                    tree.reset_variables()
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
                        print(tree.metadata)
                        print(original_tree)
                        print(decoded)
                        print(str(pfin))
                        print(remove_wiki)
                        raise exc

                    try:
                        # Although we use penman to parse AMR, `smatch` uses its own AMR parser
                        # so we also verify that our linearization/delinearization works with that
                        amr_parsed = AMR.parse_AMR_line(delinearized_penman_str)

                        if amr_parsed is None:
                            raise Exception("Error parsing AMR with amr.AMR library")
                    except Exception as exc:
                        print(tree.metadata)
                        print(tree)
                        print(delinearized_penman_str)
                        print(delinearized_tree)
                        print(str(pfin))
                        print(remove_wiki)
                        raise exc

                    if original_tree != delinearized_tree:
                        print(tree.metadata)
                        print("PENMAN", penman_str)
                        print("LINEARIZED", penmanstr2linearized(penman_str))
                        print("ENCODED", encoded)
                        print("DECODED", decoded)
                        print(original_tree)
                        print(delinearized_tree)
                        print(str(pfin))
                        print(remove_wiki)
                        raise ValueError("Tree mismatch between original tree and delinearized tree")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Brute-force testing of AMR tokenization by tokenizing a linearized "
                                                  "tree, tokenizing it, decoding it, and reconstructing the tree."
                                                  " Then checking if the original and reconstructed trees are equal."
                                                  " This script is a naive, brute-force way of testing this. All .txt"
                                                  " files in a given directory will be (recursively) tested.")
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. All .txt files"
                                       " will be tested.")
    cargs = cparser.parse_args()
    main(cargs.indir)
