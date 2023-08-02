""""""
from enum import StrEnum, auto
from os import PathLike
from pathlib import Path
from typing import List, Union

import pandas as pd
import penman
from datasets import Dataset, DatasetDict
from ftfy import fix_text
from multi_amr.data.linearization import do_remove_wiki
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tqdm import tqdm


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
    src_langs: List[str],
    output_dir: Union[str, PathLike],
    dedupe: bool = False,
    remove_wiki: bool = False,
    lang: str = "en",
):
    """Given a directory of AMR files, deduplicate all files so that every file contains unique files. We also process
     the text for the sake of normalization. This is needed because the AMR3.0 corpus sometimes has unexpected
     characters or encodings, sometimes the sentences are tokenized and sometimes they are not, and the punctuation
     marks are inconsistent. So for the text, the data is ftfy.fix_text, punctuation normalized, detokenized. The AMR
     is only ftfy and punctuation normalized.

    :param amr_dirs: dirs with sub directories train, training, dev, valid, and or test. For multilingual datasets, you
     can enter multiple paths to their respective paths. Make sure to specify their respecitve language with 'src_langs'
    :param src_langs: languages associated, in order, with 'amr_dirs'
    :param output_dir: dir to write the new structure and files to with fixed sentences
    :param dedupe: whether to deduplicate the data. This will ensure that there will be no duplicates within and
     between splits
    :param remove_wiki: whether to remove wiki from the AMR entries
    :param lang: which language code is the text in (not the AMR)? This will be used for normalization
    """

    pdout = Path(output_dir).resolve()
    pdout.mkdir(exist_ok=True, parents=True)
    punct_norm_amr = MosesPunctNormalizer(lang="en")
    punct_norm_text = MosesPunctNormalizer(lang=lang)
    detokenizer = MosesDetokenizer(lang=lang)
    detokenize = detokenizer.detokenize
    data = {"metadata": [], "sentence": [], "penmanstr": [], "split_type": [], "src_lang": []}

    for amr_dir, src_lang in zip(amr_dirs, src_langs):
        pdin = Path(amr_dir).resolve()
        for psplit in pdin.glob("*"):
            if not psplit.is_dir():
                continue

            split_type = SplitType.from_string(psplit.stem)

            for pfin in tqdm(list(psplit.glob("*.txt")), unit="file", desc=split_type):
                with pfin.open(encoding="utf-8") as fhin:
                    for tree in penman.iterparse(fhin):
                        tree.reset_variables()
                        metadata = {**tree.metadata, "src_lang": src_lang}
                        data["metadata"].append(metadata)
                        sentence = tree.metadata["snt"]
                        # 1. Fix text
                        # 2. Normalize punctuation (e.g. NLLB does not support en-dashes)
                        # 3. Detokenize
                        sentence = punct_norm_text.normalize(fix_text(sentence))
                        sentence = detokenize(sentence.split())
                        data["sentence"].append(sentence)
                        # It seems that some AMR is unparseable in some cases due to metadata (?; e.g. in test set)
                        # so we empty the metadata before continuing
                        tree.metadata = {}

                        # NOTE: the fix_text is important to make sure the reference tree also is correctly formed,
                        # e.g. (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                        # 1. Fix
                        # 2. Normalize punctuations -- note that this will remove indentation from the penman str,
                        # but it should still work correctly when parsed with penman
                        penman_str = punct_norm_amr.normalize((fix_text(penman.format(tree))))
                        if remove_wiki:
                            penman_str = do_remove_wiki(penman_str)
                        data["penmanstr"].append(penman_str)
                        data["split_type"].append(split_type)
                        data["src_lang"].append(src_lang)

    df = pd.DataFrame(data)
    del data

    if dedupe:
        df_len_before = len(df.index)
        df.drop_duplicates(subset=["sentence"], inplace=True)
        print(f"Dropped {(df_len_before-len(df.index)):,} duplicates!")

    datasets = DatasetDict()
    for split_type, groupdf in df.groupby("split_type"):
        groupdf.drop(columns=["split_type"], inplace=True)
        datasets[split_type] = Dataset.from_pandas(groupdf)
        print(f"Processed {split_type} set containing {len(groupdf):,} samples!")

    datasets.save_to_disk(pdout)


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
        required=True,
    )
    cparser.add_argument(
        "-s",
        "--src_langs",
        nargs="+",
        help="languages associated, in order, with 'amr_dirs'",
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
        "--lang",
        default="en",
        help="which language code is the text in (not the AMR)? This will be used for normalization.",
    )

    cargs = cparser.parse_args()
    prepare_dataset(**vars(cargs))


if __name__ == "__main__":
    main()
