from typing import List, Union

import torch
from transformers import PreTrainedTokenizer

from mbart_amr.data.tokens import ROLES


def ends_in_valid_role(tokenizer: PreTrainedTokenizer, input_ids: Union[List[int], torch.LongTensor]):
    """We are checking whether the input_ids end in a valid role, which may end in a number"""
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    if tokenizer.convert_ids_to_tokens(input_ids[-1]) in ROLES:  # If the last item is a role, valid
        return True
    else:
        while input_ids:
            token_idx = input_ids.pop(-1)

            if is_number(tokenizer.decode(token_idx)):  # If the last item is a number, continue
                continue
            elif tokenizer.convert_ids_to_tokens(token_idx) in ROLES:  # If we encounter a role, valid
                return True
            else:  # All other cases, False
                return False

    return False


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

