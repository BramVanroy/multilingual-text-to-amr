def test_ends_in_prep(tokenizer):
    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref2 :endrel", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is False

    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref2", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is False

    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is True

    inputs = tokenizer(":startrel evidence :sense1 :prep-", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is False

    inputs = tokenizer(":startrel bury :sense1 :prep-by", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is True

    inputs = tokenizer(":prep-by", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is True

    inputs = tokenizer(":prep-by :startrel", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_prep(inputs) is False


def test_ends_in_ref(tokenizer):
    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref2 :endrel", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_ref(inputs) is False

    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref2", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_ref(inputs) is True

    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref24", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_ref(inputs) is True

    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref2454", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_ref(inputs) is True

    inputs = tokenizer(":ref24", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_ref(inputs) is True

    inputs = tokenizer(":ref42 :ARG1", add_special_tokens=False).input_ids
    assert tokenizer.ends_in_ref(inputs) is False
