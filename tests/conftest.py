from pytest import fixture

from mbart_amr.constraints import FirstTokenProcessor, OpenCloseTokenProcessor
from mbart_amr.constraints.allowed import AllowedTokensProcessor
from mbart_amr.data.tokenization import AMRMBartTokenizer


@fixture
def tokenizer():
    return AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")


def get_firsttoken_processor(tokenizer: AMRMBartTokenizer, max_length: int):
    return FirstTokenProcessor(tokenizer, max_length, debug=True)


def get_openclose_processor(tokenizer: AMRMBartTokenizer, max_length: int):
    return OpenCloseTokenProcessor(tokenizer, max_length, debug=True)


def get_allowed_processor(tokenizer: AMRMBartTokenizer, max_length: int):
    return AllowedTokensProcessor(tokenizer, max_length, debug=True)
