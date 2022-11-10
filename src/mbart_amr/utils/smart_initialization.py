import torch
from mbart_amr.data.tokenization import AMRMBartTokenizer
from transformers import (MBartForConditionalGeneration, PreTrainedModel,
                          PreTrainedTokenizerBase)


def smart_initialization(model: MBartForConditionalGeneration, tokenizer: PreTrainedTokenizerBase):
    """Inspired by SPRING's implementation. We use noise in range -0.01, +0.01 though.
    :param model: the model whose added tokens to initialize
    :param tokenizer: the tokenizer, which also contains the new tokens
    :return: the model with updated weights for the added tokens
    """
    # Vocab size if the size without the added tokens, so the first added token is at
    # index=vocab_size
    for token_id, token in enumerate(tokenizer.added_tokens_encoder, tokenizer.vocab_size):
        if token.startswith(":"):
            if token == ":startrel":
                components = [tokenizer.bos_token]
            elif token == ":endrel":
                components = [tokenizer.eos_token]
            # str -> int -> str to normalize 01 -> 1
            elif token.startswith(":op"):
                components = ["relation", "operator", str(int(token[3:]))]
            elif token.startswith(":snt"):
                components = ["relation", "sentence", str(int(token[4:]))]
            elif token.startswith(":ARG"):
                components = ["relation", "argument", str(int(token[4:]))]
            elif token.startswith(":ref"):
                components = ["reference", str(int(token[4:]))]
            elif token.startswith(":sense"):
                components = ["meaning", str(int(token[6:]))]
            elif token == ":negation":
                components = ["not"]
            elif token == ":year2":  # make explicit, otherwise it ends up as ["year2"]
                components = ["year"]
            elif token == ":prep-":
                components = ["by"]  # random preposition
            elif token == ":conj-":
                components = ["frame"]  # random unspecified conjunction "like"
            else:
                components = ["relation"] + token.lstrip(":").split("-")
        else:
            if token == "-91":
                components = ["frame"]
            elif token == "~~of":
                components = ["relation", "of"]
            elif token == "amr-unknown":
                components = ["?"]
            elif token == "amr-choice":
                components = ["or"]
            elif token == "multi-sentence":
                components = [tokenizer.bos_token]
            elif token == "amr_XX":
                components = ["en_XX"]
            else:
                components = token.split("-")

        # Filter empty strings, possible after split
        components = " ".join([c for c in components if c])
        components_ids = tokenizer.encode(components, add_special_tokens=False)
        components_vector = torch.stack(
            [model.model.shared.weight.data[idx].clone() for idx in components_ids], dim=0
        ).mean(dim=0)
        noise = torch.FloatTensor(components_vector).uniform_(-0.01, +0.01)
        components_vector = components_vector + noise
        model.model.shared.weight.data[token_id] = components_vector + noise

    return model


def freeze_encoder(model: MBartForConditionalGeneration):
    """Freeze the encoder of the MBART model
    :param model: MBART model
    :return: the MBART model with frozen encoder (but in fact the freezing happens in-place!)
    """
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    return model
