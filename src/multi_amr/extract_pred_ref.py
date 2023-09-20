"""Extracts predictions and references from the TSV files that we created with the evaluate_amr script."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from transformers import (
    HfArgumentParser,
)


def extract(
    predictions_tsv: str,
):
    pfin = Path(predictions_tsv)
    df = pd.read_csv(pfin, encoding="utf-8", sep="\t")
    predictions = df["penman_str"].tolist()
    references = df["reference_penman_str"].tolist()

    pfpreds = pfin.with_stem(f"{pfin.stem}-preds-only").with_suffix(".txt")
    pfrefs = pfin.with_stem(f"{pfin.stem}-refs-only").with_suffix(".txt")

    pfpreds.write_text("\n\n".join(predictions))
    pfrefs.write_text("\n\n".join(references))


@dataclass
class ScriptArguments:
    predictions_tsv: str = field(metadata={"help": "the dinput tsv file with predictions"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    extract(predictions_tsv=script_args.predictions_tsv)


if __name__ == "__main__":
    main()

