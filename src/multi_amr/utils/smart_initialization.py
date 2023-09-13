import torch
from multi_amr.data.additional_tokens import AMR_TOKEN
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
        token = token.lstrip(tok_wrapper.token_prefix)
        if token == "</rel>":
            components = [" )"]  # No space
        elif token == "<rel>":
            components = [" ("]  # Space
        elif token.startswith("<pointer:") and token.endswith(">"):
            components = [" pointer", f" {int(token.split(':')[1].strip('>'))}"]
        elif token == "</lit>":
            components = ['"', ' "', " quote"]
        elif token == "<lit>":
            components = ['"', ' "', " quote"]
        elif token == "-of":
            components = [" of"]
        elif token == "amr-unknown":
            components = ["?", " who", " what", " why", " where", " when"]
        elif token == "amr-choice":
            components = [" or"]
        elif token == AMR_TOKEN:  # AMR is most similar to English
            if tok_wrapper.tokenizer_type == TokenizerType.MBART:
                components = ["en_XX"]
            elif tok_wrapper.tokenizer_type == TokenizerType.NLLB:
                components = ["eng_Latn"]
            elif tok_wrapper.tokenizer_type in (TokenizerType.T5, TokenizerType.BLOOM):
                components = [" English"]  # Not a special lang token, so add space
            elif tok_wrapper.tokenizer_type == TokenizerType.BART:
                # TODO: maybe also BOS/special token for T5?
                components = tok_wrapper.tokenizer.bos_token

            else:
                raise NotImplementedError(f"Tokenizer type {tok_wrapper.tokenizer_type} not implemented yet.")
        elif token.startswith(":"):
            if token == ":op":
                components = [" relation", " operator", f" {int(token[3:])}"]
            elif token == ":snt":
                components = [" relation", " sentence", f" {int(token[4:])}"]
            elif token == ":ARG":
                components = [" relation", " argument", f" {int(token[4:])}"]
            elif token == ":quant":
                components = [" relation", " quantity"]
            elif token == ":li":
                components = [" relation", " list", " enumeration"]
            elif token == ":ord":
                components = [" 1", " 2", " 3", " 4", " 5", " 6"]
            elif token == ":year2":  # make explicit, otherwise it ends up as ["year2"]
                components = [" year"]
            elif token == ":prep-":
                components = [" by", " in", " near", " on", " at", " with"]  # random prepositions
            elif token == ":conj-as-if":
                components = [" as-if", " as if"]
            elif token == ":negation":
                components = [" no", " not"]
            else:
                components = [" relation"] + [
                    f" {splittoken}" for splittoken in token.lstrip(":").split("-") if splittoken.strip()
                ]
        else:
            components = [f" {splittoken}" for splittoken in token.split("-")]
            # Turn "01" into "1"
            if (
                len(components) > 1 and token[-1].isdigit()
            ):  # There was a dash, so it's likely that it ends in a number
                try:
                    int_ending = int(components[-1])
                except ValueError:
                    pass
                else:
                    components[-1] = f" {int_ending}"

        components = [c for c in components if c.strip()]
        components_ids = [tok_wrapper.tokenizer.encode(item, add_special_tokens=False) for item in components]

        if noise_range:
            components_vector = torch.FloatTensor(model.config.hidden_size).uniform_(-noise_range, +noise_range)
        else:
            components_vector = torch.zeros(model.config.hidden_size)

        # Use average across every component. So for "relation, list", first create an averaged repr
        # for "relation" (from, e.g., "rel", "at", "ion" vectors) and the same for "list", and then average those
        if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.BART, TokenizerType.NLLB):
            components_vector += torch.stack(
                [
                    torch.stack([model.model.shared.weight.data[idx].clone() for idx in comp_ids]).mean(dim=0)
                    for comp_ids in components_ids
                ]
            ).mean(dim=0)
            model.model.shared.weight.data[token_id] = components_vector
        elif tok_wrapper.tokenizer_type in (TokenizerType.T5,):
            components_vector += torch.stack(
                [
                    torch.stack([model.shared.weight.data[idx].clone() for idx in comp_ids]).mean(dim=0)
                    for comp_ids in components_ids
                ]
            ).mean(dim=0)
            model.shared.weight.data[token_id] = components_vector
        elif tok_wrapper.tokenizer_type == TokenizerType.BLOOM:
            components_vector += torch.stack(
                [
                    torch.stack([model.transformer.word_embeddings.weight.data[idx].clone() for idx in comp_ids]).mean(
                        dim=0
                    )
                    for comp_ids in components_ids
                ]
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
