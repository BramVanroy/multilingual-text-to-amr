import torch
from transformers import LogitsProcessor

from multi_amr.constraints.first_token import FirstTokenProcessor
from multi_amr.constraints.open_close import OpenCloseTokenProcessor
from multi_amr.data.tokenization import AMRTokenizerWrapper


class AMRLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: AMRTokenizerWrapper, max_length: int, debug: bool = False):
        self.first_token_constraint = FirstTokenProcessor(tokenizer, max_length=max_length, debug=debug)
        self.open_close_constraint = OpenCloseTokenProcessor(tokenizer, max_length=max_length, debug=debug)

        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = self.first_token_constraint(input_ids, scores)
        scores = self.open_close_constraint(input_ids, scores)
        return scores
