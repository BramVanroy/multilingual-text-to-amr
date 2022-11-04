from pathlib import Path
from typing import List, Union, Optional

from transformers import MBartTokenizer
from ftfy import fix_text

from amr_bart.amr_bart.linearization import escape_xml

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
    )
    return out_string


class AMRMBartTokenizer(MBartTokenizer):
    # TODO: when encoding, make sure we account for escaped characters. So first unescape, i.e.,
    #  linearized = fix_text(unescape_xml(linearizer.linearized))
    # TODO: add a pre/postprocessing step so that we can convert the special tokens in the additions file to our XML tags
    @classmethod
    def from_pretrained(cls, *args, new_tokens_file: Optional[str] = None, **kwargs):
        inst = super().from_pretrained(*args, **kwargs)

        new_tokens_file = new_tokens_file if new_tokens_file else Path(__file__).resolve().parent.parent.joinpath("data/vocab.bv-additions.txt")
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
        return text
