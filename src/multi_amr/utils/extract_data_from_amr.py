from os import PathLike
from pathlib import Path
from typing import Union

from tqdm import tqdm


def get_split_name(pfin: Path):
    for split_name in ("test", "dev", "train"):
        if split_name in pfin.name:
            return split_name

    raise ValueError("Unexpected path. Expected path that contains dev, test or train.")


def extract_text_from_amr(pfin: Path):
    lines = []
    with pfin.open(encoding="utf-8") as fhin:
        for line in fhin:
            if line.startswith("# ::snt "):
                lines.append(line[8:].strip())
    return lines


def main(*indirs: Union[str, PathLike], dout: Union[str, PathLike]):
    pdout = Path(dout).resolve()
    pdout.mkdir(exist_ok=True, parents=True)

    ft_datasets = {"train": [], "dev": [], "test": []}
    for indir in tqdm(indirs, unit="language", position=0):
        pdir = Path(indir).resolve()

        if not pdir.exists():
            raise ValueError(f"Path {str(pdir)} does not exist")

        amrs_dir_name = pdir.parent.name
        lang = amrs_dir_name.split("_")[-1] if "_" in amrs_dir_name else "en"
        lang_dir = pdout.joinpath(lang)
        lang_dir.mkdir(exist_ok=True, parents=True)

        split_dirs = [p for p in pdir.glob("*") if p.is_dir() and p.name in {"dev", "test", "training"}]
        for split_dir in tqdm(split_dirs, unit="split", leave=False, position=1):
            split_name = get_split_name(split_dir)
            with lang_dir.joinpath(f"{lang}_{split_name}.txt").open("w", encoding="utf-8") as fhout:
                for pfin in tqdm(list(split_dir.glob("*.txt")), unit="file", leave=False, position=2):
                    sentences = extract_text_from_amr(pfin)
                    fhout.write("\n".join(sentences) + "\n")
                    ft_datasets[split_name].extend(sentences)

    for split_name, sentences in ft_datasets.items():
        text = "\n".join(sentences) + "\n"
        pdout.joinpath(f"merged-{split_name}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(
        r"D:\corpora\amr_annotation_3.0\data\amrs\amrs_fixed\split",
        r"D:\corpora\amr_annotation_3.0\data\amrs\amrs_nl\split",
        r"D:\corpora\amr_annotation_3.0\data\amrs\amrs_es\split",
        r"D:\corpora\amr_annotation_3.0\data\amrs\amrs_ga\split",
        dout=r"D:\corpora\amr_annotation_3.0\data\text",
    )
