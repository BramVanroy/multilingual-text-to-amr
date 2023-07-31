from pathlib import Path
from tqdm import tqdm
import penman
from ftfy import fix_text

from multi_amr.constraints.open_close import AMRLogitsProcessor
from multi_amr.data.linearization import penmantree2linearized, do_remove_wiki
from multi_amr.data.tokenization import AMRTokenizerWrapper
from multi_amr.utils import debug_build_ids_for_labels, can_be_generated


def main(indir: str, max_length: int = 1024, debug: bool = False):
    pdin = Path(indir)

    tokenizer = AMRTokenizerWrapper.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    logitsprocessor = AMRLogitsProcessor(tokenizer, max_length, debug=debug)

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
                    linearized = penmantree2linearized(original_tree)
                    input_ids = debug_build_ids_for_labels(linearized, tokenizer)
                    assert can_be_generated(input_ids, logitsprocessor, tokenizer, max_length, verbose=True)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Brute-force testing of AMR generation. We test whether the"
                                                  " constraints allow the generation of the whole test set. All .txt"
                                                  " files in a given directory will be (recursively) tested.")
    cparser.add_argument("indir", help="Input directory with AMR data. Will be recursively traversed. All .txt files"
                                       " will be tested.")
    cargs = cparser.parse_args()
    main(cargs.indir)
