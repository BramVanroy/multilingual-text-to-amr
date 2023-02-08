from mbart_amr.utils import can_be_generated, debug_build_ids_for_labels
from tests.conftest import get_firsttoken_processor


def test_starting_token(tokenizer):
    # None of the special tokens can start the sequence except for :ref1
    input_ids = debug_build_ids_for_labels(":ref1 poet :name :startrel", tokenizer)
    logitsprocessor = get_firsttoken_processor(tokenizer, 6)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 6)

    input_ids = debug_build_ids_for_labels("poet :name :startrel", tokenizer)
    logitsprocessor = get_firsttoken_processor(tokenizer, 5)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

    input_ids = debug_build_ids_for_labels(":name :startrel", tokenizer)
    logitsprocessor = get_firsttoken_processor(tokenizer, 4)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

    input_ids = debug_build_ids_for_labels(":startrel", tokenizer)
    logitsprocessor = get_firsttoken_processor(tokenizer, 3)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 3)

    input_ids = debug_build_ids_for_labels("<s>", tokenizer)
    logitsprocessor = get_firsttoken_processor(tokenizer, 3)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 3)
