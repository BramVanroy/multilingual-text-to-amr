import logging
from collections import Counter
from math import ceil
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

import penman
import torch
from ftfy import fix_text
from torch.nn.utils.rnn import pad_sequence

from mbart_amr.data.linearization import do_remove_wiki
from mbart_amr.data.tokenization import AMRMBartTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


def collate_amr(
        samples: List[dict],
        tokenizer: AMRMBartTokenizer,
):
    """Collate a given batch from the dataset by padding up its given values.

    :param tokenizer: modified AMR tokenizer to use
    :param samples: a given batch
    :return: a dictionary with keys such as input_ids and labels, with values tensors
    """
    encoded = tokenizer.pad({"input_ids": [s["input_ids"] for s in samples],
                             "attention_mask": [s["attention_mask"] for s in samples]},
                            pad_to_multiple_of=4,
                            return_tensors="pt")

    # Pad with -100 so that loss functions ignore the padding tokens
    labels = pad_sequence([torch.LongTensor(s["labels"]) for s in samples], padding_value=-100, batch_first=True)

    return {**encoded, "labels": labels}


class AMRDataset(Dataset):
    """
    :param input_max_seq_length: optional max sequence length to truncate the input data to
    :param output_max_seq_length: optional max sequence length to truncate the output data (labels) to
    """

    def __init__(
            self,
            dins: List[Union[str, PathLike]],
            src_langs: List[str],
            tokenizer: AMRMBartTokenizer,
            remove_wiki: bool = False,
            input_max_seq_length: Optional[int] = None,
            output_max_seq_length: Optional[int] = None,
            max_samples_per_language: Optional[int] = None,
    ):
        if src_langs is None or len(dins) != len(src_langs):
            raise ValueError(
                "The number of input directories (one per language) is not the same as the number of given"
                " source languages. These have to be the same. Make sure that the given source languages"
                " are language codes that are compatible with the model that you use."
            )

        self.pdins = [Path(din) for din in dins]
        self.src_langs = src_langs
        self.remove_wiki = remove_wiki
        self.max_samples_per_language = max_samples_per_language

        self.sentences = []
        self.penmanstrs = []
        self.metadatas = []
        self.encoded_inputs = []
        self.labels = []

        for src_lang, pdin in zip(self.src_langs, self.pdins):
            if not pdin.exists():
                raise FileNotFoundError(f"The given directory, {str(pdin)}, was not found.")
            tokenizer.src_lang = src_lang
            n_samples = 0
            for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
                with pfin.open(encoding="utf-8") as fhin:
                    for tree in penman.iterparse(fhin):
                        tree.reset_variables()
                        # NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
                        # (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                        penman_str = fix_text(penman.format(tree))
                        if self.remove_wiki:
                            penman_str = do_remove_wiki(penman_str)
                        self.sentences.append(tree.metadata["snt"])
                        self.penmanstrs.append(penman_str)
                        metadata = {**tree.metadata, "src_lang": src_lang}
                        self.metadatas.append(metadata)

                        encoded_inputs = tokenizer(
                            tree.metadata["snt"],
                            truncation=True,
                            max_length=input_max_seq_length,
                            return_tensors="pt",
                        )
                        encoded_inputs = {k: v[0] for k, v in encoded_inputs.items()}
                        self.encoded_inputs.append(encoded_inputs)
                        labels = tokenizer.encode_penmanstrs(
                            penman_str,
                            truncation=True,
                            max_length=output_max_seq_length,
                            return_tensors="pt",
                        ).input_ids[0]

                        self.labels.append(labels)

                        n_samples += 1

                        if self.max_samples_per_language and n_samples == self.max_samples_per_language:
                            break

                if self.max_samples_per_language and n_samples == self.max_samples_per_language:
                    break

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return {
            **self.encoded_inputs[idx],
            "id": idx,
            "sentence": self.sentences[idx],
            "penmanstr": self.penmanstrs[idx],
            "metadata": self.metadatas[idx],
            "labels": self.labels[idx]
        }
