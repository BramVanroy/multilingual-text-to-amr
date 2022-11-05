from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import re

from transformers import MBartTokenizer, BatchEncoding
from ftfy import fix_text

from amr_bart.amr_bart.linearization import escape_xml, Linearizer, unescape_xml
from amr_bart.amr_bart.prepostprocessor import linearized2inputstr, inputstr2linearized


# Idea Tim: remove outlier from special tokens, so that these outliers will be automatically tokenized
# This may make it difficult to postprocess the trees, but we can try!

def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

    Args:
        out_string (`str`): The text to clean up.

    Returns:
        `str`: The cleaned-up string.
    """
    out_string = (
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
        .replace(" -of", "-of")
        .replace(" -91", "-91")
        .replace(" -quantity", "-quantity")
        .replace(" -entity", "-entity")
    )

    # AMR specific
    # Prepositions
    out_string = re.sub(r":prep-\s+(\w+)", r"prep-\1", out_string)
    # :ARG150 will be tokenized as :ARG1 50. So the second capturing group might be present as part of the first token
    # Add extra space afterwords to make sure that we separate the special token from any following token
    out_string = re.sub(r":(ARG|op|snt|sense|term|tref)(\d+)?\s+(\d+)", r":\1\2\3 ", out_string)
    # Clean-up whitespaces
    out_string = " ".join(out_string.split())

    return out_string


class AMRMBartTokenizer(MBartTokenizer):
    # TODO: when encoding, make sure we account for escaped characters. So first unescape, i.e.,
    #  linearized = fix_text(unescape_xml(linearizer.linearized))
    # TODO: add a pre/postprocessing step so that we can convert the special tokens in the additions file to our XML tags
    @classmethod
    def from_pretrained(cls, *args, new_tokens_file: Optional[str] = None, **kwargs) -> AMRMBartTokenizer:
        inst = super().from_pretrained(*args, **kwargs)

        new_tokens_file = new_tokens_file if new_tokens_file else Path(__file__).resolve().parent.parent.joinpath("data/vocab/additions.txt")
        tokens_to_add = set([token for token in new_tokens_file.read_text(encoding="utf-8").splitlines() if token])

        voc = set(inst.get_vocab().keys())
        new_tokens = tokens_to_add - voc
        if new_tokens:
            inst.add_tokens(list(sorted(new_tokens)))
            print(f"Added {len(new_tokens):,} new tokens!")

        return inst

    def decode_and_escape(self, token_ids: List[int]):
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        filtered_tokens = [token if token in self.added_tokens_encoder else escape_xml(fix_text(token)) for token in
                           filtered_tokens]
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
        text = inputstr2linearized(text)

        return text

    def _linearize_and_unescape(self, penman_str: str):
        """Given a penman AMR string, linearize the tree, unescape potential escpaed characters (e.g. &amp;China&amp;)
        and convert the linearized format to a generic token format.

        The linearized format is XML-compatible, which is useful when converting to/from AMR/
        The generic token format is more "generic" than the initial linearized format, in that it specifies some
        hard-coded, specific tokens (e.g. :ARG0) but also allows generic tokens for outliers or exceptional cases:
        ":ARG26" can be tokenized with the added token ":ARG" and the existing token "26". Therefore, this generic token
        format is more suitable for language modeling/tokenization than the initial linearized form.
        """
        linearizer = Linearizer.from_penman_str(penman_str)
        linearizer.penman_tree.reset_variables()

        # Unescape so that the model does not have to predict things like &amp;China&amp;
        linearized = unescape_xml(linearizer.linearized)

        # Convert Linearized pseudo-XML to manageable special tokens
        prepared = linearized2inputstr(linearized)
        return prepared

    def encode_penman(
        self,
        penman_str: str,
        **kwargs
    ) -> List[int]:
        """Given one or  penman AMR strings, linearize them and then encode them with the tokenizer to get input_ids.
        See: _linearize_and_unescape() """
        prepared_str = self._linearize_and_unescape(penman_str)
        return super().encode(prepared_str, **kwargs)

    def encode_penman_plus(
        self,
        penman_strs: Union[str, List[str]],
        **kwargs
    ) -> BatchEncoding:
        """Given one or  penman AMR strings, linearize them and then encode them with the tokenizer to get input_ids
        as well as other important items such as attention masks.

        See: _linearize_and_unescape() """
        if isinstance(penman_strs, str):
            penman_strs = [penman_strs]

        prepared_strs = [self._linearize_and_unescape(penman_str) for penman_str in penman_strs]
        return super().encode_plus(prepared_strs, **kwargs)
