from transformers import AutoConfig


def get_model(model_name_or_path, from_pretrained=True):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout

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
                noise.uniform_(-0.1, +0.1)
                model.model.shared.weight.data[idx] = vec + noise
                modified += 1
