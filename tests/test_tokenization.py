from pathlib import Path
from tqdm import tqdm
import penman
from ftfy import fix_text

from amr_bart.amr_bart.linearization import penmanstr2linearized, linearized2penmantree
from amr_bart.amr_bart.tokenization_amr_bart import AMRMBartTokenizer


def main(indir: str):
    pdin = Path(indir)
    tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="amr_XX")
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        with pfin.open(encoding="utf-8") as fhin:
            for tree in penman.iterparse(fhin):
                tree.reset_variables()
                # NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
                # (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                penman_str = fix_text(penman.format(tree))
                original_tree = penman.parse(penman_str)

                encoded = tokenizer.encode_penmanstr(penman_str, remove_wiki=False)
                decoded = tokenizer.decode_and_fix(encoded)

                try:
                    delinearized_tree = linearized2penmantree(decoded)
                except penman.exceptions.DecodeError as exc:
                    print(original_tree.metadata)
                    print(original_tree)
                    print(decoded)
                    raise exc

                if original_tree != delinearized_tree:
                    print(tree.metadata)
                    print("PENMAN", penman_str)
                    print("LINEARIZED", penmanstr2linearized(penman_str))
                    print("ENCODED", encoded)
                    print("DECODED", decoded)
                    print(original_tree)
                    print(delinearized_tree)
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
