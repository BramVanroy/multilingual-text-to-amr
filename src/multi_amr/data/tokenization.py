from __future__ import annotations

import logging
import re
from typing import List, Tuple, Union

import numpy as np
import torch
from ftfy import fix_text
from multi_amr.data.linearization import penmanstr2linearized, tokenize_except_quotes
from multi_amr.data.tokens import (
    AMR_LANG_CODE,
    CHOICE,
    ENDLIT,
    ENDREL,
    MULTI_SENTENCE,
    OF_SUFFIX,
    OTHER_ROLES,
    PREP_PREFIX,
    STARTLIT,
    STARTREL,
    TOKENS_TO_ADD,
    UNKOWN, SUFFIXES,
)
from multi_amr.utils import is_number
from tqdm import tqdm
from transformers import BatchEncoding, MBartTokenizer, T5Tokenizer

logger = logging.getLogger(__name__)


def clean_up_tokenization(out_string: str) -> str:
    """Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    :param out_string: the text to clean up
    :return: the cleaned-up string
    """
    out_string = (
        # In original BART implementation
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        # AMR specific
        .replace(f" {OF_SUFFIX}", OF_SUFFIX)
        .replace(" -91", "-91")
        .replace(" -quantity", "-quantity")
        .replace(" -entity", "-entity")
    )

    # AMR specific
    # Generic prepositions/conunctions, e.g. `:prep-by` or `:conj-as-if`
    out_string = re.sub(r":(prep|conj)-\s+(\w+)", r":\1-\2", out_string)
    # Merging e.g. :ARG1 2 into :ARG12. But only if the next token is a :startrel or :startlit and not
    # any other relation (starting with :)
    out_string = re.sub(rf":(ARG|op|snt)\s*(\d+)?\s+(\d+)\s*({OF_SUFFIX})? (?:(?!:)|(?=:startrel|:startlit|:ref))", r":\1\2\3\4 ", out_string)

    # Adding space before/after :startlit/:endlit
    out_string = re.sub(r"\s*:(startlit|endlit)\s*", r" :\1 ", out_string)

    # Merging e.g. :ref1 2 into :ref12
    out_string = re.sub(r":(ref)\s*(\d+)?\s+(\d+)", r":\1\2\3", out_string)

    # Clean-up whitespaces
    out_string = " ".join(out_string.split())

    return out_string


class AMRMBartTokenizer(MBartTokenizer):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inst = super().from_pretrained(*args, **kwargs)

        # Only add tokens that are not in the vocabulary.py yet
        tokens_to_add = set(TOKENS_TO_ADD)
        voc = set(inst.get_vocab().keys())
        new_tokens = list(sorted(tokens_to_add - voc))

        if new_tokens:
            inst.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new tokens to tokenizer")

        # Just adding amr_XX to voc and defining it here as the tgt_lang is not enough
        # because we are always just calling the tokenizer (as if it were the source tokenizer)
        # However, we cannot even use it as a target tokenizer with tgt_lang amr_XX, because
        # the MBARTTokenizer only allows special language codes as tgt_lang for this purpose so
        # we cannot take that approach. Instead we will be replacing the special source language
        # token in "encode_penmanstrs" with our own, amr_XX one
        inst.amr_token = AMR_LANG_CODE
        inst.tgt_lang = inst.amr_token  # AMR is always target in our case
        inst.voc_size = len(inst)

        # single idx with type: int
        inst.amr_token_idx = inst.convert_tokens_to_ids(inst.amr_token)
        inst.start_rel_idx = inst.convert_tokens_to_ids(STARTREL)
        inst.end_rel_idx = inst.convert_tokens_to_ids(ENDREL)
        inst.start_lit_idx = inst.convert_tokens_to_ids(STARTLIT)
        inst.end_lit_idx = inst.convert_tokens_to_ids(ENDLIT)
        inst.lang_idx = inst.convert_tokens_to_ids(AMR_LANG_CODE)
        inst.multisent_idx = inst.convert_tokens_to_ids(MULTI_SENTENCE)
        inst.of_idx = inst.convert_tokens_to_ids(OF_SUFFIX)
        inst.prep_idx = inst.convert_tokens_to_ids(PREP_PREFIX)
        inst.unknown_idx = inst.convert_tokens_to_ids(UNKOWN)
        inst.choice_idx = inst.convert_tokens_to_ids(CHOICE)

        # multiple idxs with type: LongTensor
        inst.rel_idxs = torch.LongTensor([inst.start_rel_idx, inst.end_rel_idx])
        inst.otherroles_idxs = torch.LongTensor(inst.convert_tokens_to_ids(OTHER_ROLES))
        inst.special_suff_idxs = torch.LongTensor(inst.convert_tokens_to_ids(SUFFIXES))

        inst.special_tokens_idxs = torch.LongTensor(inst.all_special_ids)
        inst.added_tokens_idxs = torch.LongTensor(list(inst.added_tokens_encoder.values()))
        inst.voc_idxs_for_mask = torch.arange(inst.voc_size)

        if isinstance(inst, MBartTokenizer):
            inst.lang_idxs = torch.LongTensor(list(inst.id_to_lang_code.keys()))
        elif isinstance(inst, T5Tokenizer):
            # T5 works with general prefixes that are part of the input, e.g. "Translate English to German: "
            # so there are no language codes
            inst.lang_idxs = None
        else:
            raise ValueError(f"Tokenizer type '{type(inst)}' not supported.")
        inst.all_special_ids_tensor = torch.LongTensor(inst.all_special_ids + [inst.amr_token_idx])

        return inst

    def decode_and_fix(
        self,
        token_ids: Union[List[List[int]], List[int], torch.Tensor, np.ndarray],
        pbar: bool = False,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Modified from the original HF Tokenizer. Note the run fix_text on the deocded tokens if they
        are not a special token, to solve some potential character differences in input and output.

        Works on both sequences or batches and therefore always returns a list.

        :param token_ids: Tensor or list of token ids, potentially with a batch dimension
        :param pbar: Whether to show a progressbar during decoding
        :return: a list of decoded AMR linearizations
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(dim=0)
        elif isinstance(token_ids, np.ndarray):
            if token_ids.ndim == 1:
                token_ids = np.expand_dims(token_ids, axis=0)
        elif isinstance(token_ids[0], int):
            token_ids = [token_ids]

        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.LongTensor(token_ids)

        linearized_amrs = []
        for ids in tqdm(token_ids, desc="Decoding", disable=not pbar):
            if skip_special_tokens:
                ids = self.remove_special_tokens(ids)

            if ids.numel() == 0:
                continue

            tokens = self.convert_ids_to_tokens(ids.tolist())
            filtered_tokens = [token if token in self.added_tokens_encoder else fix_text(token) for token in tokens]

            # To avoid mixing byte-level and unicode for byte-level BPT
            # we need to build string separately for added tokens and byte-level tokens
            # cf. https://github.com/huggingface/transformers/issues/1133
            sub_texts = []
            current_sub_text = []
            for token in filtered_tokens:
                if token in self.added_tokens_encoder:
                    if current_sub_text:
                        sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                        current_sub_text = []
                    sub_texts.append(token)
                else:
                    current_sub_text.append(token)
            if current_sub_text:
                sub_texts.append(self.convert_tokens_to_string(current_sub_text))

            text = " ".join(sub_texts)
            text = clean_up_tokenization(text)

            linearized_amrs.append(text)

        return linearized_amrs

    def encode_penmanstrs(
        self, penman_strs: Union[str, List[str]], remove_wiki: bool = True, **kwargs
    ) -> BatchEncoding:
        """Given one or more penman AMR strings, linearize them and then encode them with the tokenizer to get input_ids
        as well as other important items such as attention masks.

        Note: padding=True, truncation=True, and return_tensors="pt" will always be enabled!
        """
        if isinstance(penman_strs, str):
            penman_strs = [penman_strs]

        prepared_strs = [penmanstr2linearized(penman_str, remove_wiki=remove_wiki) for penman_str in penman_strs]
        encoded = self(prepared_strs, **kwargs, padding=True, truncation=True, return_tensors="pt")

        # We need to replace the final language token. Currently this is hard to implement in the HF model because
        # they use special language tokens for the language that you cannot easily modify
        # So we just do it as a post-processing step here: replacing the last token by the amr_XX ID
        input_ids = encoded["input_ids"]
        # Replace all the language IDs with the amr_token_id
        if self.lang_idxs is not None:
            input_ids[torch.isin(input_ids, self.lang_idxs)] = self.amr_token_idx

        return encoded

    def remove_special_tokens(self, input_ids: torch.LongTensor):
        """NOTE: only removes special tokens and amr_XX, NOT the added tokens"""

        # Because amr_XX is not a real "special token", it does not get ignored so we have to remove it ourselves
        # It is included in all_special_ids_tensor
        return input_ids[~torch.isin(input_ids, self.all_special_ids_tensor)]
