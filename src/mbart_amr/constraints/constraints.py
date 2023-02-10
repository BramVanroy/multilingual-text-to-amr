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
