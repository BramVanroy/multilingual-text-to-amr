import torch
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from transformers import PreTrainedModel


def smart_initialization(model: PreTrainedModel, tok_wrapper: AMRTokenizerWrapper, noise_range: float = 0.1):
    """Initialize the new tokens based on their content, i.e. averaging the embeddings of its components

    :param model: the model whose added tokens to initialize
    :param tok_wrapper: the tok_wrapper wrapper containing the tok_wrapper, already augmented with new tokens
    :param noise_range: we add noise to the tokens that are similar to other tokens from a uniform distribution
    that spans [-noise_range, +noise_range]. The default is the default noise used in SPRING
    :return: the model with updated weights for the added tokens
    """
    tokenizer = tok_wrapper.tokenizer
    # Vocab size if the size without the added tokens, so the first added token is at
    # index=vocab_size
    for token_id, token in enumerate(tok_wrapper.added_vocab.keys(), tokenizer.vocab_size):
        # IMPORTANT: MANUALLY ADD SPACES TO COMPONENTS THAT NEED THEM.
        # ')', FOR INSTANCE DOES NOT NEED A STARTING SPACE BECAUSE WE DO NOT TYPICALLY ENCOUNTER THAT IN FLOWING TEXT
        # TODO: maybe re-add spaces where useful
        token = token.lstrip(tok_wrapper.token_prefix)
        if token == "</rel>":
            components = [")"]  # No space
        elif token == "<rel>":
            components = ["("]  # Space
        elif token.startswith("<pointer:") and token.endswith(">"):
            components = ["pointer", f"{int(token.split(':')[1].strip('>'))}"]
        # elif token == "</lit>":
        #     components = ['"', "quote"]
        # elif token == "<lit>":
        #     components = ['"', "quote"]
        # elif token == "</of>":
        #     components = ["-of"]
        # elif token == "amr-unknown":
        #     components = ["?", "who", "what", "why", "where", "when"]
        # elif token == "amr-choice":
        #     components = ["or"]
        # elif token == "multi-sentence":
        #     components = ["relation", "paragraph", "sentences", "list"]
        elif token == "<AMR>":  # AMR is most similar to English
            if tok_wrapper.tokenizer_type == TokenizerType.MBART:
                components = ["en_XX"]
            elif tok_wrapper.tokenizer_type == TokenizerType.NLLB:
                components = ["eng_Latn"]
            elif tok_wrapper.tokenizer_type in (TokenizerType.T5, TokenizerType.BLOOM, TokenizerType.BART):
                components = ["English"]  # Not a special lang token, so add space
            else:
                raise NotImplementedError(f"Tokenizer type {tok_wrapper.tokenizer_type} not implemented yet.")
        elif token.startswith(":"):
            if token == ":op":
                components = ["relation", "operator", f"{int(token[3:])}"]
            elif token == ":snt":
                components = ["relation", "sentence", f"{int(token[4:])}"]
            elif token == ":ARG":
                components = ["relation", "argument", f"{int(token[4:])}"]
            # elif token == ":quant":
            #     components = ["relation", "quantity"]
            # elif token == ":li":
            #     components = ["relation", "list", "enumeration"]
            # elif token == ":ord":
            #     components = ["relation", "ordinal"]
            # elif token == ":mod":
            #     components = ["relation", "modify", "change", "adjective", "modifier"]
            elif token == ":negation":
                components = ["not", "no"]
            # elif token == ":year2":  # make explicit, otherwise it ends up as ["year2"]
            #     components = ["year"]
            # elif token == ":prep-":
            #     components = ["by", "in", "near", "on", "at", "with"]  # random prepositions
            # elif token == ":conj-as-if":
            #     components = ["as-if", "as if"]
            else:
                components = ["relation"] + [
                    f"{splittoken}" for splittoken in token.lstrip(":").split("-") if splittoken.strip()
                ]
        else:
            components = [f"{splittoken}" for splittoken in token.split("-")]
            # TODO: re-enable
            # if (
            #     len(components) > 1 and token[-1].isdigit()
            # ):  # There was a dash, so it's likely that it ends in a number
            #     try:
            #         int_ending = int(components[-1])
            #     except ValueError:
            #         pass
            #     else:
            #         components[-1] = f"{int_ending}"

        # TODO: maybe remove the following block until next comment. This is the original SPRING
        # Prefix AFTER the item. This is a bug but also present in original SPRING.
        components_ids = []
        for item in components:
            if f"{item}{tok_wrapper.token_prefix}" in tokenizer.get_vocab():
                ids = tokenizer.convert_tokens_to_ids([f"{item}{tok_wrapper.token_prefix}"])
                print("MATCH!", token, item)
            else:
                ids = tok_wrapper.tokenizer.encode(item, add_special_tokens=False)
            components_ids.append(ids)
        components_ids = [item for ids in components_ids for item in ids]  # Flatten

        vecs = []
        for token_id in components_ids:
            vec_split = model.model.shared.weight.data[token_id].clone()
            vecs.append(vec_split)

        if vecs:
            vec = torch.stack(vecs, 0).mean(0)
            noise = torch.empty_like(vec)
            noise.uniform_(-0.1, +0.1)
            model.model.shared.weight.data[token_id] = vec + noise

        # Filter empty strings, possible after split
        # TODO: this is NOT the spring implementation, but instead takes micro averages of the componenent vectors
        # components = [c for c in components if c.strip()]
        # components_ids = [tok_wrapper.tokenizer.encode(item, add_special_tokens=False) for item in components]
        #
        # if noise_range:
        #     components_vector = torch.FloatTensor(model.config.hidden_size).uniform_(-noise_range, +noise_range)
        # else:
        #     components_vector = torch.zeros(model.config.hidden_size)
        #
        # if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.BART, TokenizerType.NLLB):
        #     components_vector += torch.stack(
        #         [
        #             torch.stack([model.model.shared.weight.data[idx].clone() for idx in comp_ids]).mean(dim=0)
        #             for comp_ids in components_ids
        #         ]
        #     ).mean(dim=0)
        #     model.model.shared.weight.data[token_id] = components_vector
        # elif tok_wrapper.tokenizer_type in (TokenizerType.T5,):
        #     components_vector += torch.stack(
        #         [
        #             torch.stack([model.shared.weight.data[idx].clone() for idx in comp_ids]).mean(dim=0)
        #             for comp_ids in components_ids
        #         ]
        #     ).mean(dim=0)
        #     model.shared.weight.data[token_id] = components_vector
        # elif tok_wrapper.tokenizer_type == TokenizerType.BLOOM:
        #     components_vector += torch.stack(
        #         [
        #             torch.stack([model.transformer.word_embeddings.weight.data[idx].clone() for idx in comp_ids]).mean(
        #                 dim=0
        #             )
        #             for comp_ids in components_ids
        #         ]
        #     ).mean(dim=0)
        #     model.transformer.word_embeddings.weight.data[token_id] = components_vector
        # else:
        #     raise NotImplementedError(f"Model with type {tok_wrapper.tokenizer_type} not implemented yet.")
    return model


def freeze_encoder(model: PreTrainedModel):
    """Freeze the encoder of a pretrained model
    :param model: pretrained model
    :return: the pretrained model with frozen encoder (but in fact the freezing happens in-place!)
    """
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    return model
