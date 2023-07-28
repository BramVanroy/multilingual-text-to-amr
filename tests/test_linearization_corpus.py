from pathlib import Path
from typing import Optional

from tqdm import tqdm
import penman
from ftfy import fix_text

from amr import AMR

from multi_amr.data.linearization import penmantree2linearized, linearized2penmanstr, do_remove_wiki


def main(indir: str, start_from: Optional[int] = None):
    pdin = Path(indir)

    tree_idx = 0
    with open("all-linearized.txt", "w", encoding="utf-8") as fhout:
        for remove_wiki in (True, False):
            for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file", desc=f"Remove wiki? {remove_wiki}"):
                with pfin.open(encoding="utf-8") as fhin:
                    for tree in penman.iterparse(fhin):
                        tree_idx += 1
                        if start_from is not None and start_from > 0 and tree_idx < start_from:
                            continue
                        tree.reset_variables()
                        # NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
                        # (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                        penman_str = fix_text(penman.format(tree))

                        if remove_wiki:
                            penman_str = do_remove_wiki(penman_str)

                        original_tree = penman.parse(penman_str)

                        linearized = penmantree2linearized(original_tree)
                        if remove_wiki:
                            fhout.write(f"{linearized}\n\n")

                        delinearized_penman_str = linearized2penmanstr(linearized)
                        delinearized_tree = penman.parse(delinearized_penman_str)

                        if original_tree != delinearized_tree:
                            print(tree.metadata)
                            print(original_tree)
                            print(delinearized_tree)
                            print(linearized)
                            raise ValueError("Tree mismatch between original tree and delinearized tree")

                        try:
                            # Although we use penman to parse AMR, `smatch` uses its own AMR parser
                            # so we also verify that our linearization/delinearization works with that
                            amr_parsed = AMR.parse_AMR_line(delinearized_penman_str)

                            if amr_parsed is None:
                                raise Exception("Error parsing AMR with amr.AMR library")
                        except Exception as exc:
                            print(tree.metadata)
                            print(original_tree)
                            print(delinearized_penman_str)
                            print(delinearized_tree)
                            raise exc



if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Brute-force testing of AMR linearization. All .txt"
                                                  " files in a given directory will be (recursively) tested.")
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. All .txt files"
                                       " will be tested.")
    cparser.add_argument("--start", type=int, help="Start processing from this tree index. Useful if you know exactly"
                                                   " where something went wrong.")
    cargs = cparser.parse_args()
    main(cargs.indir, cargs.start)
