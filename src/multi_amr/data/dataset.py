import logging
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

import penman
import torch
from ftfy import fix_text
from multi_amr.data.linearization import do_remove_wiki
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from multi_amr.data.tokens import AMR_LANG_CODE
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
    tok_wrapper: AMRTokenizerWrapper,
    input_max_seq_length: Optional[int] = None,
    output_max_seq_length: Optional[int] = None,
):
    """Collate a given batch from the dataset by 1. tokenizing a given sentence and getting its attention mask,
    token_ids, etc. for input; 2. linearizing and tokenizing the associated penman str as the labels.

    :param tok_wrapper: modified AMR tokenizer wrapper to use
    :param input_max_seq_length: optional max sequence length to truncate the input data to
    :param output_max_seq_length: optional max sequence length to truncate the output data (labels) to
    :param samples: a given batch
    :return: a dictionary with keys such as input_ids and labels, with values tensors
    """
    src_langs = Counter([s["metadata"]["src_lang"] for s in samples])
    src_lang = src_langs.most_common(1)[0][0]
    if len(src_langs.keys()) > 1:
        logging.warning(
            "This given batch consists of multiple source language. Therefore, the tok_wrapper will"
            f" append a single language code ({src_lang}) that is not applicable to all samples, which may"
            f" lead to poor performance."
        )

    # Set the source lang to the main language in this batch so that the correct token can be added (not used by T5)
    tok_wrapper.tokenizer.src_lang = src_lang
    # T5 uses prefixes
    task_prefix = (
        f"translate {src_lang} to {AMR_LANG_CODE}: " if tok_wrapper.tokenizer_type == TokenizerType.T5 else ""
    )

    encoded_inputs = tok_wrapper(
        [task_prefix + s["sentence"] for s in samples],
        padding=True,
        truncation=True,
        max_length=input_max_seq_length,
        return_tensors="pt",
    )
    num_penman_samples = len([s["penmanstr"] for s in samples if s["penmanstr"]])

    if num_penman_samples:
        labels = tok_wrapper.encode_penmanstrs(
            [s["penmanstr"] for s in samples],
            max_length=output_max_seq_length,
        ).input_ids
        labels = torch.where(labels == tok_wrapper.tokenizer.pad_token_id, -100, labels)

        return {**encoded_inputs, "labels": labels}
    else:
        return {**encoded_inputs}


class AMRDataset(Dataset):
    def __init__(
        self,
        dins: List[Union[str, PathLike]],
        src_langs: List[str],
        remove_wiki: bool = False,
        max_samples_per_language: Optional[int] = None,
        is_predict: bool = False,
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
            for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file", desc=f"Processing data for {src_lang}"):
                with pfin.open(encoding="utf-8") as fhin:
                    if is_predict:
                        for line in fhin:
                            self.sentences.append(line.strip())
                            metadata = {"src_lang": src_lang}
                            self.metadatas.append(metadata)
                            n_samples += 1
                            if self.max_samples_per_language and n_samples == self.max_samples_per_language:
                                break
                    else:
                        for tree in penman.iterparse(fhin):
                            tree.reset_variables()
                            metadata = {**tree.metadata, "src_lang": src_lang}
                            self.metadatas.append(metadata)
                            self.sentences.append(tree.metadata["snt"])
                            # It seems that some AMR is unparseable in some cases due to metadata (?; e.g. in test set)
                            # so we empty the metadata before continuing
                            tree.metadata = {}

                            # NOTE: the fix_text is important to make sure the reference tree also is correctly formed,
                            # e.g. (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                            penman_str = fix_text(penman.format(tree))
                            if self.remove_wiki:
                                penman_str = do_remove_wiki(penman_str)
                            self.penmanstrs.append(penman_str)

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
            "penmanstr": self.penmanstrs[idx] if idx < len(self.penmanstrs) else None,  # No penmans in predict-mode
            "metadata": self.metadatas[idx],
        }
