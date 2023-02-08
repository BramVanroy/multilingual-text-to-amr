from collections import defaultdict
from typing import Dict

import torch
from mbart_amr.data.tokenization import AMRMBartTokenizer
from mbart_amr.data.tokens import (ENDLIT, ENDREL, AMR_LANG_CODE, OF_SUFFIX, REFS,
                                   SENSES, STARTLIT, STARTREL, MULTI_SENTENCE, UNKOWN, CHOICE)
from transformers import LogitsProcessor


class AMRLogitsProcessorBase(LogitsProcessor):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int, debug: bool = False):
        self.tokenizer = tokenizer
        self.debug = debug

        # We need -1 because there is another logitprocessor (? somewhere)
        # that ensures that the last token is EOS, so we account for that
        self.max_length = max_length - 1
        self.voc_size = len(tokenizer)

        # single idx with type: int
        self.start_rel_idx = tokenizer.convert_tokens_to_ids(STARTREL)
        self.end_rel_idx = tokenizer.convert_tokens_to_ids(ENDREL)
        self.start_lit_idx = tokenizer.convert_tokens_to_ids(STARTLIT)
        self.end_lit_idx = tokenizer.convert_tokens_to_ids(ENDLIT)
        self.end_lit_idx = tokenizer.convert_tokens_to_ids(ENDLIT)
        self.lang_idx = tokenizer.convert_tokens_to_ids(AMR_LANG_CODE)
        self.multisent_idx = tokenizer.convert_tokens_to_ids(MULTI_SENTENCE)
        self.of_idx = tokenizer.convert_tokens_to_ids(OF_SUFFIX)
        self.unknown_idx = tokenizer.convert_tokens_to_ids(UNKOWN)
        self.choice_idx = tokenizer.convert_tokens_to_ids(CHOICE)

        # multiple idxs with type: LongTensor
        self.sense_idxs = torch.LongTensor(tokenizer.convert_tokens_to_ids(SENSES))
        self.ref_idxs = torch.LongTensor(tokenizer.convert_tokens_to_ids(REFS))
        self.special_tokens_idxs = torch.LongTensor(self.tokenizer.all_special_ids)
        self.added_tokens_idxs = torch.LongTensor(list(self.tokenizer.added_tokens_encoder.values()))
        self.voc_idxs_for_mask = torch.arange(self.voc_size)

        super().__init__()

    def _debug_decode(self, input_ids, skip_special_tokens=False):
        return self.tokenizer.decode_and_fix(input_ids, skip_special_tokens=skip_special_tokens)


def input_ids_counts(inputs: torch.LongTensor) -> Dict[int, int]:
    # -- collect unique counts in the current inputs for each token ID
    uniq_counts = defaultdict(int)  # Counter that will default to 0 for unknown keys
    # torch.unique returns a tuple, the sorted inp tensor, and a tensor with the frequencies
    # then, map these tensors to a list and zip them into a dict to get {input_id: frequency}
    # By updating the `counter`, we have a frequency dictionary and default values of 0
    uniq_counts.update(dict(zip(*map(torch.Tensor.tolist, inputs.unique(return_counts=True)))))

    return uniq_counts
