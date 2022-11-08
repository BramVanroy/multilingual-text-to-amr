from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

from mbart_amr.data.linearization import penmanstr2linearized
from ftfy import fix_text
from transformers import BatchEncoding, MBartTokenizer


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
        inst = MBartTokenizer.from_pretrained(*args, **kwargs)

        new_tokens_file = (
            new_tokens_file
            if new_tokens_file
            else Path(__file__).resolve().parent.parent.joinpath("data/vocab/additions.txt")
        )
        tokens_to_add = set([token for token in new_tokens_file.read_text(encoding="utf-8").splitlines() if token])

        voc = set(inst.get_vocab().keys())
        new_tokens = tokens_to_add - voc
        if new_tokens:
            inst.add_tokens(list(sorted(new_tokens)))
            print(f"Added {len(new_tokens):,} new tokens!")

        inst.tgt_lang = "amr_XX"

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

        See: _linearize_and_unescape()"""
        if isinstance(penman_strs, str):
            penman_strs = [penman_strs]

        prepared_strs = [penmanstr2linearized(penman_str, remove_wiki=remove_wiki) for penman_str in penman_strs]
        return self(prepared_strs, **kwargs)


def postprocess(text: str) -> str:
    """
    :param text:
    :return:
    """
    pass