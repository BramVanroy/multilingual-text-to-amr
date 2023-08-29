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
        token = token.lstrip(tok_wrapper.token_prefix)
        if token.startswith(":"):
            if token == "</rel>":
                components = [
                    "relation",
                    "end",
                    "</div>",
                ]  # Bold (?) assumption that the model was trained on a bit of HTML data
            elif token == "<rel>":
                components = ["relation", "start", "<div>"]  # BOS is never used in MBART
            elif token == "</lit>":
                components = ["relation", "end", "literal", '"', "quote"]
            elif token == "<lit>":
                components = ["relation", "start", "literal", '"', "quote"]
            elif token == ":op":
                components = ["relation", "operator"]
            elif token == ":snt":
                components = ["relation", "sentence"]
            elif token == ":ARG":
                components = ["relation", "argument"]
            elif token == ":quant":
                components = ["relation", "quantity"]
            elif token == ":li":
                components = ["relation", "list", "enumeration"]
            elif token == ":ord":
                components = ["relation", "ordinal"]
            elif token == ":mod":
                components = ["relation", "modify", "change", "adjective", "modifier"]
            elif token.startswith("<pointer:") and token.endswith(">"):
                components = ["reference", "like", "similar", "pointer"]
            elif token == ":negation":
                components = ["not", "no"]
            elif token == ":year2":  # make explicit, otherwise it ends up as ["year2"]
                components = ["year"]
            elif token == ":prep-":
                components = ["by", "in", "near", "on", "at", "with"]  # random prepositions
            elif token == ":conj-as-if":
                components = ["as-if", "as if"]
            else:
                components = ["relation"] + token.lstrip(":").split("-")
        else:
            if token == "</of>":
                components = ["relation", "of", "have"]
            elif token == "amr-unknown":
                components = ["?", "who", "what", "why", "where", "when"]
            elif token == "amr-choice":
                components = ["or"]
            elif token == "multi-sentence":
                components = ["relation", "paragraph", "sentences", "list"]
            elif token == "<AMR>":  # AMR is most similar to English
                if tok_wrapper.tokenizer_type == TokenizerType.MBART:
                    components = ["en_XX"]
                elif tok_wrapper.tokenizer_type == TokenizerType.NLLB:
                    components = ["eng_Latn"]
                elif tok_wrapper.tokenizer_type in (TokenizerType.T5, TokenizerType.BLOOM):
                    components = ["English"]
                else:
                    raise NotImplementedError(f"Tokenizer type {tok_wrapper.tokenizer_type} not implemented yet.")
            else:
                components = token.split("-")

        # Only keep non-empty items. Might occur when splitting, e.g. "-quantity"
        components = [item for item in components if item]

        if noise_range:
            components_vector = torch.FloatTensor(model.config.hidden_size).uniform_(-noise_range, +noise_range)
        else:
            components_vector = torch.zeros(model.config.hidden_size)
        # Filter empty strings, possible after split
        components = " ".join([c for c in components if c])
        components_ids = tokenizer.encode(components, add_special_tokens=False)

        if tok_wrapper.tokenizer_type in [TokenizerType.MBART, TokenizerType.NLLB]:
            components_vector += torch.stack(
                [model.model.shared.weight.data[idx].clone() for idx in components_ids], dim=0
            ).mean(dim=0)
            model.model.shared.weight.data[token_id] = components_vector
        elif tok_wrapper.tokenizer_type == TokenizerType.T5:
            components_vector += torch.stack(
                [model.shared.weight.data[idx].clone() for idx in components_ids], dim=0
            ).mean(dim=0)

            model.shared.weight.data[token_id] = components_vector
        elif tok_wrapper.tokenizer_type == TokenizerType.BLOOM:
            components_vector += torch.stack(
                [model.transformer.word_embeddings.weight.data[idx].clone() for idx in components_ids], dim=0
            ).mean(dim=0)

            model.transformer.word_embeddings.weight.data[token_id] = components_vector
        else:
            raise NotImplementedError(f"Model with type {tok_wrapper.tokenizer_type} not implemented yet.")

    return model


def freeze_encoder(model: PreTrainedModel):
    """Freeze the encoder of a pretrained model
    :param model: pretrained model
    :return: the pretrained model with frozen encoder (but in fact the freezing happens in-place!)
    """
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    return model
