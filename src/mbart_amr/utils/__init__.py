from collections import defaultdict
from typing import Dict

import torch
from transformers import LogitsProcessor, MBartTokenizer


def is_number(maybe_number_str: str) -> bool:
    """Check whether a given string is a number. We do not consider special cases such as 'infinity' and 'nan',
    which technically are floats. We do consider, however, floats like '1.23'.
    :param maybe_number_str: a string that might be a number
    :return: whether the given number is indeed a number
    """
    if maybe_number_str in ["infinity", "nan", "inf"]:
        return False

    try:
        float(maybe_number_str)
        return True
    except ValueError:
        return False


def can_be_generated(
    input_ids: torch.LongTensor,
    logitsprocessor: LogitsProcessor,
    tokenizer: MBartTokenizer,
    max_length: int,
    verbose: bool = False,
):
    fake_scores = torch.FloatTensor(1, len(tokenizer)).uniform_(-20, +20)
    for idx in range(1, min(input_ids.size(1), max_length)):
        scores = logitsprocessor(input_ids[:, :idx], fake_scores.clone())
        next_idx = input_ids[0, idx]

        if torch.isinf(scores[0, next_idx]):
            if verbose:
                print(
                    f"NOT POSSIBLE: {debug_decode(input_ids, tokenizer)}\nAfter {debug_decode(input_ids[:, :idx], tokenizer)},\n    {debug_decode(input_ids[:, idx], tokenizer)} was not allowed"
                )
            return False

    return True


def debug_decode(input_ids: torch.LongTensor, tokenizer: MBartTokenizer, skip_special_tokens: bool = False):
    return tokenizer.decode_and_fix(input_ids, skip_special_tokens=skip_special_tokens)[0]


def debug_build_ids_for_labels(linearized: str, tokenizer: MBartTokenizer):
    return tokenizer(f"amr_XX {linearized}", add_special_tokens=False, return_tensors="pt").input_ids

def input_ids_counts(inputs: torch.LongTensor) -> Dict[int, int]:
    # -- collect unique counts in the current inputs for each token ID
    uniq_counts = defaultdict(int)  # Counter that will default to 0 for unknown keys
    # torch.unique returns a tuple, the sorted inp tensor, and a tensor with the frequencies
    # then, map these tensors to a list and zip them into a dict to get {input_id: frequency}
    # By updating the `counter`, we have a frequency dictionary and default values of 0
    uniq_counts.update(dict(zip(*map(torch.Tensor.tolist, inputs.unique(return_counts=True)))))

    return uniq_counts
