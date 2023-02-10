from __future__ import annotations

import logging
import re
from typing import List, Union

import numpy as np
import torch
from ftfy import fix_text

from mbart_amr.data.linearization import penmanstr2linearized
from tqdm import tqdm
from transformers import BatchEncoding, MBartTokenizer
from mbart_amr.data.tokens import (AMR_LANG_CODE, CHOICE, ENDLIT, ENDREL,
                                   MULTI_SENTENCE, OF_SUFFIX, REFS, SENSES,
                                   STARTLIT, STARTREL, UNKOWN, TOKENS_TO_ADD)
from mbart_amr.utils import input_ids_counts


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
        .replace(" ~~of", "~~of")
        .replace(" -91", "-91")
        .replace(" -quantity", "-quantity")
        .replace(" -entity", "-entity")
    )

    # AMR specific
    # Generic prepositions/conunctions, e.g. `:prep-by` or `:conj-as-if`
    out_string = re.sub(r":(prep|conj)-\s+(\w+)", r":\1-\2", out_string)
    # Merging e.g. :ARG1 2 into :ARG12. But only if the next token is a :startrel or :startlit and not
    # any other relation (starting with :)
    out_string = re.sub(r":(ARG|op|snt)(\d+)?\s+(\d+) (?:(?!:)|(?=:startrel|:startlit))", r":\1\2\3 ", out_string)

    # Adding space before/after :startlit/:endlit
    out_string = re.sub(r":(startlit|endlit)", r" :\1 ", out_string)

    # Merging e.g. :sense1 2 into :sense12
    out_string = re.sub(r":(sense|ref)(\d+)\s+(\d+)", r":\1\2\3", out_string)
    # To account for phone numbers like 512-386-91 45 where -91 is incorrectly used as a special token
    # I have not found a way to 1. use -91 as a generic token while also 2. have it work well in arbitrary
    # cases like the phone number.
    out_string = re.sub(r"-91 (\d+)", r"-91\1", out_string)
    # Clean-up whitespaces
    out_string = " ".join(out_string.split())

    return out_string


class AMRMBartTokenizer(MBartTokenizer):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inst = super().from_pretrained(*args, **kwargs)

        # Only add tokens that are not in the vocabulary yet
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
        inst.end_lit_idx = inst.convert_tokens_to_ids(ENDLIT)
        inst.lang_idx = inst.convert_tokens_to_ids(AMR_LANG_CODE)
        inst.multisent_idx = inst.convert_tokens_to_ids(MULTI_SENTENCE)
        inst.of_idx = inst.convert_tokens_to_ids(OF_SUFFIX)
        inst.unknown_idx = inst.convert_tokens_to_ids(UNKOWN)
        inst.choice_idx = inst.convert_tokens_to_ids(CHOICE)

        # multiple idxs with type: LongTensor
        inst.sense_idxs = torch.LongTensor(inst.convert_tokens_to_ids(SENSES))
        inst.ref_idxs = torch.LongTensor(inst.convert_tokens_to_ids(REFS))
        inst.special_tokens_idxs = torch.LongTensor(inst.all_special_ids)
        inst.added_tokens_idxs = torch.LongTensor(list(inst.added_tokens_encoder.values()))
        inst.voc_idxs_for_mask = torch.arange(inst.voc_size)
        inst.lang_idxs = torch.LongTensor(list(inst.id_to_lang_code.keys()))
        inst.all_special_ids_tensor = torch.LongTensor(inst.all_special_ids + [inst.amr_token_idx])

        return inst


    def decode_and_fix(
        self,
        token_ids: Union[List[List[int]], List[int], torch.Tensor, np.ndarray],
        pbar: bool = False,
        skip_special_tokens: bool = True
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
            ids = self.postprocess(ids)
            ids = self.convert_ids_to_tokens(ids)

            filtered_tokens = [
                token if token in self.added_tokens_encoder else fix_text(token) for token in ids
            ]

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
        input_ids[torch.isin(input_ids, self.lang_ids)] = self.amr_token_id

        return encoded

    def remove_special_tokens(self, input_ids: torch.LongTensor):
        """NOTE: only removes special tokens and amr_XX, NOT the added tokens
        """

        # Because amr_XX is not a real "special token", it does not get ignored so we have to remove it ourselves
        # It is included in all_special_ids_tensor
        return input_ids[~torch.isin(input_ids, self.all_special_ids_tensor)]


    def postprocess(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        uniq_counts = input_ids_counts(input_ids)
        if uniq_counts[self.start_lit_idx] != uniq_counts[self.end_lit_idx]:
            input_ids = torch.cat((input_ids, torch.LongTensor([self.end_lit_idx])))

        rel_start_end_diff = uniq_counts[self.start_rel_idx] - uniq_counts[self.end_rel_idx]
        if rel_start_end_diff:
            input_ids = torch.cat((input_ids, torch.LongTensor([self.end_rel_idx] * rel_start_end_diff)))

        return input_ids