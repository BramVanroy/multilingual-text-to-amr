from mbart_amr.utils import can_be_generated, debug_build_ids_for_labels
from tests.conftest import get_allowed_processor


def test_start_rel_follow_role(tokenizer):
    # :start_rel can only follow valid role
    input_ids = debug_build_ids_for_labels("poet :startrel :endrel", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 5)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

    input_ids = debug_build_ids_for_labels("poet :name :startrel", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 5)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

def test_ref_follow_start_or_role(tokenizer):
    # :refXX can only follow :start_rel or a role...
    input_ids = debug_build_ids_for_labels(":ARG0 :startrel :ref1", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 5)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

    input_ids = debug_build_ids_for_labels(":ARG0 :ref1", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

    input_ids = debug_build_ids_for_labels(":sense1 :ref1", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

    input_ids = debug_build_ids_for_labels("cookie :ref1", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

    # ... except for :ref1 which can occur at the start of the sequence
    input_ids = debug_build_ids_for_labels(":ref1", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 3)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 3)


def test_sense_follow_nonaddedspecial(tokenizer):
    # :senseXX can only follow non-added and non-special tokens
    input_ids = debug_build_ids_for_labels("over :sense3", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 3)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 3)

    input_ids = debug_build_ids_for_labels(":startrel :sense3", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 3)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 3)

    input_ids = debug_build_ids_for_labels("<s> :sense3", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 3)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 3)

def test_of_follows_role(tokenizer):
    # ~~of can only follow roles (but not sents) or preps
    input_ids = debug_build_ids_for_labels(":ARG12~~of", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 10)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 10)

    input_ids = debug_build_ids_for_labels(":prep-on-behalf~~of", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 10)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 10)

    input_ids = debug_build_ids_for_labels(":snt1 ~~of", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 3)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 3)

    # ~~of cannot follow ~~of
    input_ids = debug_build_ids_for_labels("~~of~~of", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

def test_following_endlit(tokenizer):
    # Only specific added tokens can follow an ending :endlit
    # ~~of, :prep-, amr-unknown, amr-choice, multi-sentence, amr_XX NOT allowed
    input_ids = debug_build_ids_for_labels(":endlit :ARG7", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

    input_ids = debug_build_ids_for_labels(":endlit potato", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 5)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

    input_ids = debug_build_ids_for_labels(":endlit amr-unknown", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

def ref_after_ref(tokenizer):
    # Ref cannot follow other ref
    input_ids = debug_build_ids_for_labels(":ref1 :ref2", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 4)

    input_ids = debug_build_ids_for_labels(":ref145 :ref2", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 5)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

    input_ids = debug_build_ids_for_labels(":ref1 :endrel", tokenizer)
    logitsprocessor = get_allowed_processor(tokenizer, 4)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 4)
