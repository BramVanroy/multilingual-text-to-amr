from __future__ import annotations

import logging
import re
from enum import StrEnum, auto
from typing import List, Union

import numpy as np
import torch
from ftfy import fix_text
from multi_amr.data.linearization import penmanstr2linearized
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
    SUFFIXES,
    TOKENS_TO_ADD,
    UNKOWN,
)
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedTokenizerBase,
    T5Tokenizer,
    T5TokenizerFast,
)


logger = logging.getLogger(__name__)


class TokenizerType(StrEnum):
    MBART = auto()
    NLLB = auto()
    T5 = auto()


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
    # Clean-up whitespaces before doing regexes (this is important!)
    out_string = " ".join(out_string.split())

    # AMR specific
    # Generic prepositions/conunctions, e.g. `:prep-by` or `:conj-as-if`
    out_string = re.sub(r":(prep|conj)-\s+(\w+)", r":\1-\2", out_string)
    # Merging e.g. :ARG1 2 into :ARG12. But only if the next token is a :startrel or :startlit and not
    # any other relation (starting with :)
    # print("BEFORE REPL", out_string)
    out_string = re.sub(
        rf":(ARG|op|snt)(\d+)\s+(\d+)\s*({OF_SUFFIX})?\s+(?:(?!:)|(?=:startrel|:startlit|:ref))",
        r":\1\2\3\4 ",
        out_string,
    )
    # print("AFTER REPL", out_string)
    # Adding space before/after :startlit/:endlit
    out_string = re.sub(r"\s*:(startlit|endlit)\s*", r" :\1 ", out_string)

    # Merging e.g. :ref1 2 into :ref12
    out_string = re.sub(r"(:ref\d+)\s+(\d+)", r"\1\2", out_string)

    # Clean-up whitespaces
    out_string = " ".join(out_string.split())

    return out_string


class AMRTokenizerWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

        # Only add tokens that are not in the vocabulary.py yet
        tokens_to_add = set(TOKENS_TO_ADD)
        voc = set(self.tokenizer.get_vocab().keys())
        new_tokens = list(sorted(tokens_to_add - voc))

        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new tokens to tok_wrapper")

        # Just adding AMR to voc and defining it here as the tgt_lang is not enough
        # because we are always just calling the tok_wrapper (as if it were the source tok_wrapper)
        # However, we cannot even use it as a target tok_wrapper with tgt_lang AMR, because
        # the MBARTTokenizer only allows special language codes as tgt_lang for this purpose so
        # we cannot take that approach. Instead we will be replacing the special source language
        # token in "encode_penmanstrs" with our own, AMR one
        self.amr_token = AMR_LANG_CODE
        self.tokenizer.voc_size = len(self.tokenizer)

        self.added_vocab = self.tokenizer.get_added_vocab()

        # single idx with type: int
        self.amr_token_idx = self.tokenizer.convert_tokens_to_ids(self.amr_token)
        self.start_rel_idx = self.tokenizer.convert_tokens_to_ids(STARTREL)
        self.end_rel_idx = self.tokenizer.convert_tokens_to_ids(ENDREL)
        self.start_lit_idx = self.tokenizer.convert_tokens_to_ids(STARTLIT)
        self.end_lit_idx = self.tokenizer.convert_tokens_to_ids(ENDLIT)
        self.lang_idx = self.tokenizer.convert_tokens_to_ids(AMR_LANG_CODE)
        self.multisent_idx = self.tokenizer.convert_tokens_to_ids(MULTI_SENTENCE)
        self.of_idx = self.tokenizer.convert_tokens_to_ids(OF_SUFFIX)
        self.prep_idx = self.tokenizer.convert_tokens_to_ids(PREP_PREFIX)
        self.unknown_idx = self.tokenizer.convert_tokens_to_ids(UNKOWN)
        self.choice_idx = self.tokenizer.convert_tokens_to_ids(CHOICE)

        # multiple idxs with type: LongTensor
        self.rel_idxs = torch.LongTensor([self.start_rel_idx, self.end_rel_idx])
        self.otherroles_idxs = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(OTHER_ROLES))
        self.special_suff_idxs = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(SUFFIXES))

        self.special_tokens_idxs = torch.LongTensor(self.tokenizer.all_special_ids)
        self.added_tokens_idxs = torch.LongTensor(list(self.added_vocab.values()))
        self.voc_idxs_for_mask = torch.arange(self.tokenizer.voc_size)

        if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            self.tokenizer_type = TokenizerType.MBART
            self.tokenizer.tgt_lang = self.amr_token  # AMR is always target in our case
            self.lang_idxs = torch.LongTensor(list(self.tokenizer.lang_code_to_id.values()))
        elif isinstance(self.tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            self.tokenizer_type = TokenizerType.NLLB
            self.tokenizer.tgt_lang = self.amr_token  # AMR is always target in our case
            self.lang_idxs = torch.LongTensor(list(self.tokenizer.lang_code_to_id.values()))
        elif isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            # T5 works with general prefixes that are part of the input, e.g. "Translate English to German: "
            # so there are no language codes
            self.tokenizer_type = TokenizerType.T5
            self.lang_idxs = None
        else:
            raise ValueError(f"Tokenizer type '{type(self.tokenizer)}' not supported.")
        self.all_special_ids_tensor = torch.LongTensor(self.tokenizer.all_special_ids + [self.amr_token_idx])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(tokenizer=AutoTokenizer.from_pretrained(*args, **kwargs))

    def decode_and_fix(
        self,
        token_ids: Union[List[List[int]], List[int], torch.Tensor, np.ndarray],
        pbar: bool = False,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Modified from the original HF Tokenizers. Note the run fix_text on the deocded tokens if they
        are not a special token, to solve some potential character differences in input and output.

        Works on both sequences or batches and therefore always returns a list.

        :param token_ids: Tensor or list of token ids, potentially with a batch dimension
        :param pbar: Whether to show a progressbar during decoding
        :param skip_special_tokens: Whether to skip special tokens, including the special AMR token
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

            tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
            filtered_tokens = [token if token in self.added_vocab else fix_text(token) for token in tokens]

            # To avoid mixing byte-level and unicode for byte-level BPT
            # we need to build string separately for added tokens and byte-level tokens
            # cf. https://github.com/huggingface/transformers/issues/1133
            sub_texts = []
            current_sub_text = []
            for token in filtered_tokens:
                if token in self.added_vocab:
                    if current_sub_text:
                        sub_texts.append(self.tokenizer.convert_tokens_to_string(current_sub_text))
                        current_sub_text = []
                    sub_texts.append(token)
                else:
                    current_sub_text.append(token)
            if current_sub_text:
                sub_texts.append(self.tokenizer.convert_tokens_to_string(current_sub_text))

            text = " ".join(sub_texts)
            text = clean_up_tokenization(text)

            linearized_amrs.append(text)

        return linearized_amrs

    def encode_penmanstrs(
        self, penman_strs: Union[str, List[str]], remove_wiki: bool = True, padding=True, truncation=True, **kwargs
    ) -> BatchEncoding:
        """Given one or more penman AMR strings, linearize them and then encode them with the tok_wrapper to get input_ids
        as well as other important items such as attention masks.

        Note: padding=True, truncation=True, and return_tensors="pt" will always be enabled!
        """
        if isinstance(penman_strs, str):
            penman_strs = [penman_strs]

        prepared_strs = [penmanstr2linearized(penman_str, remove_wiki=remove_wiki) for penman_str in penman_strs]
        encoded = self.tokenizer(prepared_strs, **kwargs, padding=padding, truncation=truncation, return_tensors="pt")

        # We need to replace the final language token. Currently this is hard to implement in the HF model because
        # they use special language tokens for the language that you cannot easily modify
        # So we just do it as a post-processing step here: replacing the last token by the AMR ID
        input_ids = encoded["input_ids"]
        # Replace all the language IDs with the amr_token_id
        if self.lang_idxs is not None:
            input_ids[torch.isin(input_ids, self.lang_idxs)] = self.amr_token_idx

        return encoded

    def remove_special_tokens(self, input_ids: torch.LongTensor):
        """NOTE: only removes special tokens and AMR, NOT the added tokens"""

        # Because `AMR_lang` is not a real "special token", it does not get ignored so we have to remove it ourselves
        # It is included in all_special_ids_tensor
        return input_ids[~torch.isin(input_ids, self.all_special_ids_tensor)]

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
