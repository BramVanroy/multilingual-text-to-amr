import torch
from mbart_amr.constraints import FirstTokenProcessor, OpenCloseTokenProcessor
from mbart_amr.constraints.allowed import AllowedTokensProcessor
from mbart_amr.data.linearization import linearized2penmanstr
from mbart_amr.data.tokenization import AMRMBartTokenizer
from transformers import (LogitsProcessor, LogitsProcessorList,
                          MBartForConditionalGeneration)


class AMRLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int, debug: bool = False):
        self.first_token_constraint = FirstTokenProcessor(tokenizer, max_length=max_length, debug=debug)
        self.open_close_constraint = OpenCloseTokenProcessor(tokenizer, max_length=max_length, debug=debug)
        self.allowed_constraint = AllowedTokensProcessor(tokenizer, max_length=max_length, debug=debug)

        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = self.first_token_constraint(input_ids, scores)
        scores = self.open_close_constraint(input_ids, scores)
        scores = self.allowed_constraint(input_ids, scores)
        return scores


def main():
    tokenizer = AMRMBartTokenizer.from_pretrained("BramVanroy/mbart-en-to-amr", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained("BramVanroy/mbart-en-to-amr")

    model.resize_token_embeddings(len(tokenizer))

    gen_kwargs = {"max_length": 36, "num_beams": 1}

    if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
        gen_kwargs["max_length"] = model.config.max_length
    gen_kwargs["num_beams"] = (
        gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else model.config.num_beams
    )
    gen_kwargs["logits_processor"] = LogitsProcessorList([AMRLogitsProcessor(tokenizer, gen_kwargs["max_length"])])

    encoded = tokenizer("Should we go home now ?", return_tensors="pt")
    generated = model.generate(
        **encoded,
        **gen_kwargs,
    )

    decoded = tokenizer.decode_and_fix(generated[0])[0]
    print(generated)
    print(decoded)
    print(linearized2penmanstr(decoded))


if __name__ == "__main__":
    main()
