from pathlib import Path
from tqdm import tqdm
import penman

from amr_bart.amr_bart.linearization import penmantree2linearized, linearized2penmantree


def main(indir: str):
    pdin = Path(indir)
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        with pfin.open(encoding="utf-8") as fhin:
            for tree in penman.iterparse(fhin):
                tree.reset_variables()

                linearized = penmantree2linearized(tree)
                delinearized_tree = linearized2penmantree(linearized)

                if tree != delinearized_tree:
                    print(tree.metadata)
                    print(tree)
                    print(delinearized_tree)
                    raise ValueError("Tree mismatch between original tree and delinearized tree")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Brute-force testing of AMR linearization. All .txt"
                                                  " files in a given directory will be (recursively) tested.")
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. All .txt files"
                                       " will be tested.")
    cargs = cparser.parse_args()
    main(cargs.indir)
