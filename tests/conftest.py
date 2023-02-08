import torch
from pytest import fixture
from transformers import LogitsProcessor

from mbart_amr.constraints import FirstTokenProcessor, OpenCloseTokenProcessor
from mbart_amr.data.tokenization import AMRMBartTokenizer


@fixture
def tokenizer():
    return AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")


def decode(input_ids: torch.LongTensor, tokenizer: AMRMBartTokenizer, skip_special_tokens: bool = False):
    return tokenizer.decode_and_fix(input_ids, skip_special_tokens=skip_special_tokens)[0]


def build_ids_for_labels(linearized: str, tokenizer: AMRMBartTokenizer):
    return tokenizer(f"amr_XX {linearized}", add_special_tokens=False, return_tensors="pt").input_ids


def get_firsttoken_processor(tokenizer: AMRMBartTokenizer, max_length: int):
    return FirstTokenProcessor(tokenizer, max_length, debug=True)


def get_openclose_processor(tokenizer: AMRMBartTokenizer, max_length: int):
    return OpenCloseTokenProcessor(tokenizer, max_length, debug=True)


def can_be_generated(input_ids: torch.LongTensor, logitsprocessor: LogitsProcessor, tokenizer: AMRMBartTokenizer,
                     max_length: int, verbose: bool = False):
    fake_scores = torch.FloatTensor(1, len(tokenizer)).uniform_(-20, +20)
    for idx in range(1, min(input_ids.size(1), max_length)):
        scores = logitsprocessor(input_ids[:, :idx], fake_scores.clone())
        next_idx = input_ids[0, idx]

        if torch.isinf(scores[0, next_idx]):
            if verbose:
                print(f"NOT POSSIBLE: {decode(input_ids, tokenizer)}\nAfter {decode(input_ids[:, :idx], tokenizer)},\n" \
                      f"    {decode(input_ids[:, idx], tokenizer)} was not allowed")
            return False

    return True
