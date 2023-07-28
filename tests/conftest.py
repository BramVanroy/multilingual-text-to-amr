from pytest import fixture

from multi_amr.data.tokenization import AMRMBartTokenizer


@fixture
def tokenizer():
    return AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
