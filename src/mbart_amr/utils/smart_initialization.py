import torch
from mbart_amr.data.tokenization import AMRMBartTokenizer
from transformers import MBartForConditionalGeneration


def smart_initialization(model: MBartForConditionalGeneration, tokenizer: AMRMBartTokenizer, noise_range: float = 0.1):
    """Initialize the new tokens based on their content, i.e. averaging the embeddings of its components

    :param model: the model whose added tokens to initialize
    :param tokenizer: the tokenizer, which also contains the new tokens
    :param noise_range: we add noise to the tokens that are similar to other tokens from a uniform distribution
    that spans [-noise_range, +noise_range]. The default is the default noise used in SPRING
    :return: the model with updated weights for the added tokens
    """
    # Vocab size if the size without the added tokens, so the first added token is at
    # index=vocab_size
    for token_id, token in enumerate(tokenizer.added_tokens_encoder, tokenizer.vocab_size):
        if token.startswith(":"):
            if token == ":endrel":
                components = ["relation", "end", tokenizer.eos_token]
            elif token == ":startrel":
                components = ["relation", "start", tokenizer.eos_token]  # BOS is never used in MBART
            elif token == ":endlit":
                components = ["relation", "end", "literal", '"']
            elif token == ":startlit":
                components = ["relation", "start", "literal", '"']
            elif token.startswith(":op"):
                components = ["relation", "operator", str(token[3:])]
            elif token.startswith(":snt"):
                components = ["relation", "sentence", str(token[4:])]
            elif token.startswith(":ARG"):
                components = ["relation", "argument", str(token[4:])]
            elif token.startswith(":ref"):
                components = ["reference", str(token[4:])]
            elif token.startswith(":sense"):
                components = ["meaning", str(token[6:])]
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
            if token == "-91":
                components = ["frame"]
            elif token == "~~of":
                components = ["relation", "of", "have"]
            elif token == "amr-unknown":
                components = ["?"]
            elif token == "amr-choice":
                components = ["or"]
            elif token == "multi-sentence":
                components = ["relation", "multiple", "sentence"]
            elif token == "amr_XX":  # AMR is most similar to English
                components = ["en_XX"]
            else:
                components = token.split("-")

        # Filter empty strings, possible after split
        components = " ".join([c for c in components if c])
        components_ids = tokenizer.encode(components, add_special_tokens=False)
        components_vector = torch.stack(
            [model.model.shared.weight.data[idx].clone() for idx in components_ids], dim=0
        ).mean(dim=0)

        if noise_range:
            noise = torch.FloatTensor(components_vector).uniform_(-noise_range, +noise_range)
            components_vector = components_vector + noise

        model.model.shared.weight.data[token_id] = components_vector

    return model


def freeze_encoder(model: MBartForConditionalGeneration):
    """Freeze the encoder of the MBART model
    :param model: MBART model
    :return: the MBART model with frozen encoder (but in fact the freezing happens in-place!)
    """
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    return model
