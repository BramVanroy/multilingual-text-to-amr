import ftfy

"""Fix all the sentences in "# ::snt " in given AMR files. Some of these contain encoding issues and need
to be fixed. Also special characters/quotation marks will be normalized. This is especially important before 
translation."""
from os import PathLike
from pathlib import Path
from typing import Union

from tqdm import tqdm


def translate(
    amr_dir: Union[str, PathLike],
    output_dir: Union[str, PathLike],
    verbose: bool = False,
):
    """Given a directory of AMR, all .txt files will recursively be traversed and sentences fixed. All the lines
    that start with "# ::snt " will be fixed.

    :param amr_dir: dir with AMR files (potentially deep-structured)
    :param output_dir: dir to write the new structure and files to with fixed sentences
    :param verbose: whether to print fixed sentences to stdout. Only the fixed sentences that differ from the original
     sentence will be printed
    """
    pdin = Path(amr_dir).resolve()
    pdout = Path(output_dir).resolve()
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        pfout = pdout / pfin.relative_to(pdin)
        pfout.parent.mkdir(exist_ok=True, parents=True)
        with pfout.open("w", encoding="utf-8") as fhout:
            lines = pfin.read_text(encoding="utf-8").splitlines()
            # get sentece lines and remove "# ::snt " prefix
            sentences = [(line_idx, line[8:]) for line_idx, line in enumerate(lines) if line.startswith("# ::snt ")]

            for sent_idx, sent in sentences:
                fixed_sentence = ftfy.fix_text(sent).replace("\n", " ")
                fixed_sentence = " ".join(fixed_sentence.split())
                if verbose and fixed_sentence != lines[sent_idx][8:]:
                    print(f"ORIG:  {lines[sent_idx][8:]}")
                    print(f"FIXED: {fixed_sentence}")
                    print()
                lines[sent_idx] = f"# ::snt {fixed_sentence}"

            fhout.write("\n".join(lines) + "\n")


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("amr_dir", help="dir with AMR files (potentially deep-structured)")
    cparser.add_argument("output_dir", help="dir to write the new structure and files to with fixed sentences")
    cparser.add_argument("-v", "--verbose", action="store_true",
                         help="whether to print fixed sentences to stdout. Only the fixed sentences that differ from"
                              " the original sentence will be printed")

    cargs = cparser.parse_args()
    translate(**vars(cargs))


if __name__ == "__main__":
    main()

"""
python src/amr_bart/utils/amr_fix.py D:\corpora\amr_annotation_3.0\data\amrs D:\corpora\amr_annotation_3.0\data\amrs_fixed -v
"""
