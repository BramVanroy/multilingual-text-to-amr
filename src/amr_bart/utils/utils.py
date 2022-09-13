import torch
from amr_bart.amr_bart.modeling_amr_bart import AMRBartForConditionalGeneration
from amr_bart.amr_bart.tokenization_amr_bart import AMRBartTokenizer
from transformers import AutoConfig


def instantiate_model_and_tokenizer(
    model_name_or_path,
    config_name_or_path,
    tokenizer_name_or_path,
    additional_tokens_smart_init=True,
    dropout=0.15,
    attention_dropout=0.15,
    from_pretrained=True,
    init_reverse=False,
    collapse_name_ops=False,
    use_pointer_tokens=False,
):
    config = AutoConfig.from_pretrained(config_name_or_path)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout

    tokenizer = AMRBartTokenizer.from_pretrained(
        tokenizer_name_or_path,
        collapse_name_ops=collapse_name_ops,
        use_pointer_tokens=use_pointer_tokens,
        config=config,
    )

    if from_pretrained:
        model = AMRBartForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
    else:
        model = AMRBartForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer.encoder))

    if additional_tokens_smart_init:
        modified = 0
        for tok, idx in tokenizer.encoder.items():
            tok = tok.lstrip(tokenizer.INIT)

            if idx < tokenizer.old_enc_size:
                continue

            elif tok.startswith("<pointer:") and tok.endswith(">"):
                tok_split = ["pointer", str(tok.split(":")[1].strip(">"))]

            elif tok.startswith("<"):
                continue

            elif tok.startswith(":"):
                if tok.startswith(":op"):
                    tok_split = ["relation", "operator", str(int(tok[3:]))]

                elif tok.startswith(":snt"):
                    tok_split = ["relation", "sentence", str(int(tok[4:]))]

                elif tok.startswith(":ARG"):
                    tok_split = ["relation", "argument", str(int(tok[4:]))]

                else:
                    tok_split = ["relation"] + tok.lstrip(":").split("-")

            else:
                tok_split = tok.split("-")

            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + tokenizer.INIT
                if s_ in tokenizer.encoder:
                    tok_split.append(s_)
                else:
                    tok_split.extend(tokenizer._tok_bpe(s))

            vecs = []
            for s in tok_split:
                idx_split = tokenizer.encoder.get(s, -1)
                if idx_split > -1:
                    vec_split = model.model.shared.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, 0.1)
                model.model.shared.weight.data[idx] = vec + noise
                modified += 1

    if init_reverse:
        model.init_reverse_model()

    return model, tokenizer
