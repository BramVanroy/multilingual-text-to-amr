from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from transformers import BatchEncoding, PreTrainedTokenizer


@dataclass
class DenoisingCollator:
    """"""
    tokenizer: PreTrainedTokenizer
    decoder_start_token_id: Optional[int] = None
    shift_func: Callable = field(default=None, init=False)
    max_length: Optional[int] = None

    def __call__(self, examples: List[Dict[str, torch.LongTensor]]) -> BatchEncoding:
        # input_ids may be shortened due to span masking, so instead get max_length from the labels
        max_length = max(len(ex["labels"]) for ex in examples) if self.max_length is None else self.max_length

        batch = self.tokenizer.pad([{"input_ids": ex["input_ids"]} for ex in examples],
                                   return_tensors="pt",
                                   return_attention_mask=False,
                                   padding="max_length",
                                   max_length=max_length)

        # Labels must be -100 for padding so crossentropyloss can ignore them
        batch["labels"] = torch.full_like(batch["input_ids"], fill_value=-100, dtype=torch.long)
        for ex_idx, ex in enumerate(examples):
            batch["labels"][ex_idx, :ex["labels"].size(0)] = ex["labels"]

        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], self.tokenizer.pad_token_id)

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()

        return batch


def shift_tokens_right(input_ids: torch.LongTensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    Different from HF, which I believe to have a bug.
    See: https://github.com/huggingface/transformers/pull/18985#issuecomment-1243657702
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined.")

    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()

    shifted = torch.full_like(input_ids, pad_token_id)
    for b_idx in range(input_ids.size(0)):
        shifted[b_idx, 1:index_of_eos[b_idx]+1] = prev_output_tokens[b_idx, :index_of_eos[b_idx]].clone()

    shifted[:, 0] = decoder_start_tokens

    return shifted