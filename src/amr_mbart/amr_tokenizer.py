def get_tokenizer(
    tokenizer_name_or_path=None,
    checkpoint=None,
    additional_tokens_smart_init=True,
    dropout=0.15,
    attention_dropout=0.15,
    from_pretrained=True,
    init_reverse=False,
    collapse_name_ops=False,
    use_pointer_tokens=False,
    raw_graph=False,
):

    tokenizer = AMRBartTokenizer.from_pretrained(
        tokenizer_name_or_path,
        collapse_name_ops=collapse_name_ops,
        use_pointer_tokens=use_pointer_tokens,
        config=config,
    )
