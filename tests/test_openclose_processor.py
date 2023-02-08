from tests.conftest import get_openclose_processor, build_ids_for_labels, can_be_generated


def test_closing_tokens(tokenizer):
    # REL: an :endrel cannot immediately follow a :startrel
    input_ids = build_ids_for_labels("poet :name :startrel :endrel", tokenizer)
    logitsprocessor = get_openclose_processor(tokenizer, 6)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 6)

    # REL: an :endrel cannot be generated if there is no current open :startrel
    input_ids = build_ids_for_labels("poet :name :endrel", tokenizer)
    logitsprocessor = get_openclose_processor(tokenizer, 5)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 5)

    input_ids = build_ids_for_labels("contrast :sense1 :ARG2 :startrel understand :endrel :endrel", tokenizer)
    logitsprocessor = get_openclose_processor(tokenizer, 9)
    assert not can_be_generated(input_ids, logitsprocessor, tokenizer, 9)

    input_ids = build_ids_for_labels("contrast :sense1 :ARG2 :startrel understand :endrel", tokenizer)
    logitsprocessor = get_openclose_processor(tokenizer, 6)
    assert can_be_generated(input_ids, logitsprocessor, tokenizer, 6)
