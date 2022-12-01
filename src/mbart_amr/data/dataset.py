import logging
from collections import Counter
from math import ceil
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

import penman
import torch
from ftfy import fix_text
from mbart_amr.data.linearization import do_remove_wiki
from mbart_amr.data.tokenization import AMRMBartTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


KEEP_KEYS = {
    "input_ids",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "head_mask",
    "decoder_head_mask",
    "cross_attn_head_mask",
    "encoder_outputs",
    "past_key_values",
    "inputs_embeds",
    "decoder_inputs_embeds",
    "labels",
}


def collate_amr(
    samples: List[dict],
    tokenizer: AMRMBartTokenizer,
    input_max_seq_length: Optional[int] = None,
    output_max_seq_length: Optional[int] = None,
):
    """Collate a given batch from the dataset by 1. tokenizing a given sentence and getting its attention mask,
    token_ids, etc. for input; 2. linearizing and tokenizing the associated penman str as the labels.

    :param tokenizer: modified AMR tokenizer to use
    :param input_max_seq_length: optional max sequence length to truncate the input data to
    :param output_max_seq_length: optional max sequence length to truncate the output data (labels) to
    :param samples: a given batch
    :return: a dictionary with keys such as input_ids and labels, with values tensors
    """
    src_langs = Counter([s["metadata"]["src_lang"] for s in samples])
    src_lang = src_langs.most_common(1)[0][0]
    if len(src_langs.keys()) > 1:
        logging.warning(
            "This given batch consists of multiple source language. Therefore, the tokenizer will"
            f" append a single language code ({src_lang}) that is not applicable to all samples, which may"
            f" lead to poor performance."
        )

    # Set the source lang to the main language in this batch so that the correct token can be added
    tokenizer.src_lang = src_lang
    encoded_inputs = tokenizer(
        [s["sentence"] for s in samples],
        padding=True,
        truncation=True,
        max_length=input_max_seq_length,
        return_tensors="pt",
    )
    labels = tokenizer.encode_penmanstrs(
        [s["penmanstr"] for s in samples],
        max_length=output_max_seq_length,
    ).input_ids
    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)

    return {**encoded_inputs, "labels": labels}


class AMRDataset(Dataset):
    def __init__(
        self,
        dins: List[Union[str, PathLike]],
        src_langs: List[str],
        remove_wiki: bool = False,
        max_samples_per_language: Optional[int] = None,
    ):
        if src_langs is None or dins is None or len(dins) != len(src_langs):
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

        for src_lang, pdin in zip(self.src_langs, self.pdins):
            if not pdin.exists():
                raise FileNotFoundError(f"The given directory, {str(pdin)}, was not found.")

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

                        n_samples += 1

                        if self.max_samples_per_language and n_samples == self.max_samples_per_language:
                            break

                if self.max_samples_per_language and n_samples == self.max_samples_per_language:
                    break

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return {
            "id": idx,
            "sentence": self.sentences[idx],
            "penmanstr": self.penmanstrs[idx],
            "metadata": self.metadatas[idx],
        }
