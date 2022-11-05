from pathlib import Path
from tqdm import tqdm
import penman

from amr_bart.amr_bart.linearization import Linearizer
from amr_bart.amr_bart.tokenization_amr_bart import AMRMBartTokenizer


def main(indir: str):
    pdin = Path(indir)
    tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="amr_XX")
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        with pfin.open(encoding="utf-8") as fhin:
            for tree in penman.iterparse(fhin):
                tree.reset_variables()
                penman_str = penman.format(tree)
                original = Linearizer.from_penman_str(penman_str)

                encoded = tokenizer.encode_penman(penman_str)
                decoded = tokenizer.decode_and_escape(encoded)
                print(decoded)
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
