from __future__ import annotations

import re
from typing import List, Optional, Union

import torch
from ftfy import fix_text
from mbart_amr.data.linearization import penmanstr2linearized
from transformers import BatchEncoding, MBartTokenizer

from mbart_amr.data.tokens import TOKENS_TO_ADD


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
    # Merging e.g. :ARG1 2 into :ARG12. But only if the next token is a :startrel and not
    # any other relation (starting with :)
    out_string = re.sub(r":(ARG|op|snt)(\d+)?\s+(\d+) (?:(?!:)|(?=:startrel))", r":\1\2\3 ", out_string)

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
    def from_pretrained(cls, *args, new_tokens_file: Optional[str] = None, **kwargs):
        inst = super().from_pretrained(*args, **kwargs)

        tokens_to_add = set(TOKENS_TO_ADD)

        voc = set(inst.get_vocab().keys())
        new_tokens = tokens_to_add - voc
        if new_tokens:
            inst.add_tokens(list(sorted(new_tokens)))
            print(f"Added {len(new_tokens):,} new tokens!")

        # Just adding amr_XX to voc and defining it here as the tgt_lang is not enough
        # because we are always just calling the tokenizer (as if it were the source tokenizer)
        # However, we cannot even use it as a target tokenizer with tgt_lang amr_XX, because
        # the MBARTTokenizer only allows special language codes as tgt_lang for this purpose so
        # we cannot take that approach. Instead we will be replacing the special source language
        # token in "encode_penmanstrs" with our own, amr_XX one
        inst.amr_token = "amr_XX"
        inst.amr_token_id = inst.convert_tokens_to_ids("amr_XX")
        inst.tgt_lang = "amr_XX"
        inst.lang_ids = torch.LongTensor(list(inst.id_to_lang_code.keys()))

        return inst

    def decode_and_fix(self, token_ids: List[int]) -> str:
        """Modified from the original HF Tokenizer. Note that run fix_text on the deocded tokens if they
        are not a special token, to solve some potential character differences in input and output.
        Note that this does not work on the batch level but on a single sequence!

        :param token_ids: List of token ids
        :return: an output string, representing the linearized graph
        """
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        # Because amr_XX is not a real "special token", it does not get ignored so we have to remove it ourselves
        filtered_tokens = [token for token in filtered_tokens if token != self.amr_token]
        filtered_tokens = [
            token if token in self.added_tokens_encoder else fix_text(token) for token in filtered_tokens
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

        return text

    def encode_penmanstrs(
            self, penman_strs: Union[str, List[str]], remove_wiki: bool = True, **kwargs
    ) -> BatchEncoding:
        """Given one or  penman AMR strings, linearize them and then encode them with the tokenizer to get input_ids
        as well as other important items such as attention masks.

        Note: padding=True, truncation=True, and return_tensors="pt" will always be enabled!

        See: _linearize_and_unescape()"""
        if isinstance(penman_strs, str):
            penman_strs = [penman_strs]

        prepared_strs = [penmanstr2linearized(penman_str, remove_wiki=remove_wiki) for penman_str in penman_strs]
        encoded = self(prepared_strs, **kwargs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")

        # We need to replace the final language token. Currently this is hard to implement in the HF model because
        # they use special language tokens for the language that you cannot easily modify
        # So we just do it as a post-processing step here: replacing the last token by the amr_XX ID
        input_ids = encoded["input_ids"]
        # Replace all the language IDs with the amr_token_id
        input_ids[torch.isin(input_ids, self.lang_ids)] = self.amr_token_id

        return encoded
