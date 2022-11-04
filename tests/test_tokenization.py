from pathlib import Path
from tqdm import tqdm
import penman
from ftfy import fix_text

from amr_bart.amr_bart.linearization import Linearizer, unescape_xml
from amr_bart.amr_bart.tokenization_amr_bart import AMRMBartTokenizer


def main(indir: str):
    pdin = Path(indir)
    tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="amr_XX")
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        with pfin.open(encoding="utf-8") as fhin:
            for tree in penman.iterparse(fhin):
                tree.reset_variables()
                tree_str = penman.format(tree)
                original = Linearizer.from_penman_str(tree_str)
                linearized = fix_text(unescape_xml(original.linearized))

                encoded = tokenizer.encode(linearized)
                decoded = tokenizer.decode_and_escape(encoded)
                delinearized = Linearizer.from_linearized(decoded)

                if original.penman_tree != delinearized.penman_tree:
                    print(tree.metadata)
                    print(original.penman_tree)
                    print(delinearized.penman_tree)
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
