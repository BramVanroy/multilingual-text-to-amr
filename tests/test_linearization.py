from pathlib import Path
from tqdm import tqdm
import penman

from amr_bart.amr_bart.linearization import Linearizer


def main():
    pdin = Path(r"D:\corpora\amr_annotation_3.0\data\amrs_fixed")
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        with pfin.open(encoding="utf-8") as fhin:
            for tree in penman.iterparse(fhin):
                tree.reset_variables()
                tree_str = penman.format(tree)
                original = Linearizer.from_penman_str(tree_str)
                linearized = original.linearized
                delinearized = Linearizer.from_linearized(linearized)

                if original.penman_tree != delinearized.penman_tree:
                    print(tree.metadata)
                    print(original.penman_tree)
                    print(delinearized.penman_tree)
                    raise ValueError("Tree mismatch between original tree and delinearized tree")


if __name__ == "__main__":
    main()