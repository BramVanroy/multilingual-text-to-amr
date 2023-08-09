import json
from enum import StrEnum, auto
from os import PathLike
from pathlib import Path
from typing import List, Union

import pandas as pd
import penman
from datasets import Dataset, DatasetDict
from ftfy import fix_text
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tqdm import tqdm

from multi_amr.data.linearization import remove_wiki_from_graph, dfs_linearize


class SplitType(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

    @classmethod
    def from_string(cls, split_str):
        if split_str in ("train", "training"):
            return cls.TRAIN
        elif split_str in ("validation", "dev"):
            return cls.VALIDATION
        elif split_str == "test":
            return cls.TEST
        raise ValueError(
            f"'{split_str}' is not a valid {cls.__name__} value. Valid options: 'train', 'training',"
            f" 'validation', 'dev', 'test'"
        )


def prepare_dataset(
    amr_dirs: List[Union[str, PathLike]],
    langs: List[str],
    output_dir: Union[str, PathLike],
    dedupe: bool = False,
    remove_wiki: bool = False,
    fix_ftfy: bool = False,
    normalize_punct: bool = False,
    detokenize: bool = False,
):
    """Given a directory of AMR files, deduplicate all files so that every file contains unique files. We also process
     the text for the sake of normalization. This is needed because the AMR3.0 corpus sometimes has unexpected
     characters or encodings, sometimes the sentences are tokenized and sometimes they are not, and the punctuation
     marks are inconsistent. So for the text, the data is ftfy.fix_text, punctuation normalized, detokenized. The AMR
     is only ftfy and punctuation normalized.

    :param amr_dirs: dirs with sub directories train, training, dev, valid, and or test. For multilingual datasets, you
     can enter multiple paths to their respective paths. Make sure to specify their respecitve language code with
     'langs'
    :param langs: language codes (e.g. 'en') of the text (not the AMR but the sentences). This will be used for
     punctuation normalization and detokenization. Add one per dir in 'amr_dirs'
    :param output_dir: dir to write the new structure and files to with fixed sentences
    :param dedupe: whether to deduplicate the data. This will ensure that there will be no duplicates within and
     between splits
    :param remove_wiki: whether to remove wiki from the AMR entries
    :param fix_ftfy: whether to fix text issues with ftfy in both the sentence and the linearized AMR
    :param normalize_punct: whether to normalize punctuation in both the sentence and the linearized AMR
    :param detokenize: whether to detokenize to sentence (not the linearized AMR)
    """

    pdout = Path(output_dir).resolve()
    pdout.mkdir(exist_ok=True, parents=True)
    punct_norm_amr = MosesPunctNormalizer(lang="en")  # Use English because AMR is most like English
    data = {"metadata": [], "sentence": [], "amr": [], "split_type": [], "src_lang_idx": []}

    for src_lang_idx, (src_lang, amr_dir) in enumerate(zip(langs, amr_dirs)):
        punct_norm_text = MosesPunctNormalizer(lang=src_lang)
        detokenizer = MosesDetokenizer(lang=src_lang)
        detokenize_func = detokenizer.detokenize

        pdin = Path(amr_dir).resolve()
        for psplit in pdin.glob("*"):
            if not psplit.is_dir():
                continue

            split_type = SplitType.from_string(psplit.stem)

            for pfin in tqdm(list(psplit.glob("*.txt")), unit="file", desc=split_type):
                with pfin.open(encoding="utf-8") as fhin:
                    for graph in penman.iterdecode(fhin):
                        if remove_wiki:
                            graph = remove_wiki_from_graph(graph)
                        linearized = " ".join(dfs_linearize(graph))
                        sentence = graph.metadata["snt"]

                        if fix_ftfy:
                            sentence = fix_text(sentence)
                            linearized = fix_text(linearized)

                        if normalize_punct:
                            sentence = punct_norm_text.normalize(sentence)
                            linearized = punct_norm_amr.normalize(linearized)

                        if detokenize:
                            sentence = detokenize_func(sentence.split())

                        data["metadata"].append(graph.metadata)
                        data["sentence"].append(sentence)
                        data["amr"].append(linearized)
                        data["split_type"].append(split_type)
                        # We just use an index, so that during training we can re-use the same dataset with different
                        # src_langs even though the data is the same. Sometimes we may need 'en_XX', other times
                        # 'English', or 'Latn_eng'. We can set that in our training config.
                        data["src_lang_idx"].append(src_lang_idx)

    df = pd.DataFrame(data)
    del data

    print("Example data")
    print(df.head(3))

    processing_info = {
        "amr_dirs": amr_dirs,
        "langs": langs,
        "dedupe": dedupe,
        "remove_wiki": remove_wiki,
        "fix_ftfy": fix_ftfy,
        "normalize_punct": normalize_punct,
        "detokenize": detokenize,
    }

    if dedupe:
        df_len_before = len(df.index)
        df.drop_duplicates(subset=["sentence"], inplace=True)
        print(f"Dropped {(df_len_before - len(df.index)):,} duplicates!")
        processing_info["num_dropped_duplicates"] = df_len_before - len(df.index)

    datasets = DatasetDict()
    processing_info["final_sizes"] = {}
    for split_type, groupdf in df.groupby("split_type"):
        groupdf.drop(columns=["split_type"], inplace=True)
        datasets[split_type] = Dataset.from_pandas(groupdf)
        print(f"Processed {split_type} set containing {len(groupdf):,} samples!")
        processing_info["final_sizes"][split_type] = len(groupdf)

    datasets.save_to_disk(pdout)
    pdout.joinpath("processing_info.json").write_text(json.dumps(processing_info, indent=4), encoding="utf-8")


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "-i",
        "--amr_dirs",
        nargs="+",
        help="dirs with sub directories train, training, dev, valid, and or test. For"
        " multilingual datasets, you can enter multiple paths to their respective"
        " paths. Make sure to specify their respecitve language with 'src_langs'",
    )
    cparser.add_argument(
        "--langs",
        nargs="+",
        help="language codes (e.g. 'en') of the text (not the AMR but the sentences). This will be used for"
        " punctuation normalization and detokenization. Add one per dir in 'amr_dirs'",
        required=True,
    )
    cparser.add_argument("-o", "--output_dir", help="dir to write the dataset files to")
    cparser.add_argument(
        "--dedupe",
        action="store_true",
        help="whether to deduplicate the data. This will ensure that there will be no duplicates within and"
        " between splits",
    )
    cparser.add_argument(
        "--remove_wiki",
        action="store_true",
        help="whether to remove wiki from the AMR entries",
    )
    cparser.add_argument(
        "--fix_ftfy",
        action="store_true",
        help="whether to fix text issues with ftfy in both the sentence and the linearized AMR",
    )
    cparser.add_argument(
        "--normalize_punct",
        action="store_true",
        help="whether to normalize punctuation in both the sentence and the linearized AMR",
    )
    cparser.add_argument(
        "--detokenize",
        action="store_true",
        help="whether to detokenize to sentence (not the linearized AMR)",
    )

    cargs = cparser.parse_args()
    prepare_dataset(**vars(cargs))


if __name__ == "__main__":
    main()
