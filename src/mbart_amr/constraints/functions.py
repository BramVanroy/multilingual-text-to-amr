from functools import lru_cache
from typing import List, Tuple, Union

import torch
from mbart_amr.data.tokenization import AMRMBartTokenizer
from mbart_amr.data.tokens import (ARGS, OF_SUFFIX, OPS, OTHER_ROLES,
                                   PREP_PREFIX, REFS, SENTS)
from mbart_amr.utils import is_number


def ends_in_role(
    input_ids: Union[List[int], torch.LongTensor], tokenizer: AMRMBartTokenizer, exclude_categories: List[str] = None
) -> bool:
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    cats = {"args": ARGS, "ops": OPS, "sents": SENTS, "preps": "", "others": ""}

    if not exclude_categories:
        exclude_categories = []
    else:
        if any(cat not in cats for cat in exclude_categories):
            raise ValueError(f"'exclude_categories' must be one of {list(cats.keys())}")

    last_idx = len(input_ids) - 1

    # If a regular role, return true (can end in ~~of)
    if not "other" in exclude_categories:
        other_idxs = tokenizer.convert_tokens_to_ids(OTHER_ROLES)
        for idx in reversed(range(len(input_ids))):
            if input_ids[idx] in other_idxs:
                return True
            elif idx == last_idx and tokenizer.convert_ids_to_tokens(input_ids[idx]) == OF_SUFFIX:
                continue

    # If a prep, return true
    if not "preps" in exclude_categories:
        if ends_in_prep(input_ids, tokenizer):
            return True

    # Check if it is a numberable role but do not include the excluded categories
    numberable_roles = [role for cat, roles in cats.items() if not cat in exclude_categories for role in roles]
    numberable_roles_idxs = tokenizer.convert_tokens_to_ids(numberable_roles)

    for idx in reversed(range(len(input_ids))):  # Iterate in reverse to always get longer subsequences
        if input_ids[idx] in numberable_roles_idxs:
            return True
        elif is_number(tokenizer.decode_and_fix(input_ids[idx:])[0]):  # If the last items form a number, continue
            continue
        elif (
            idx == last_idx and tokenizer.convert_ids_to_tokens(input_ids[idx]) == OF_SUFFIX
        ):  # The very last token can be ~~of
            continue
        else:
            return False

    return False


def ends_in_ref(input_ids: Union[List[int], torch.LongTensor], tokenizer: AMRMBartTokenizer) -> bool:
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    ref_idxs = tokenizer.convert_tokens_to_ids(REFS)
    for idx in reversed(range(len(input_ids))):  # Iterate in reverse to always get longer subsequences
        # if we ultimately reach a :ref\d, then this must be valid because we only got here if the
        # the later tokens were numbers
        if input_ids[idx] in ref_idxs:
            return True
        elif is_number(tokenizer.decode_and_fix(input_ids[idx:])[0]):  # If the last items form a number, continue
            continue
        else:
            return False

    return False


def ends_in_prep(input_ids: Union[List[int], torch.LongTensor], tokenizer: AMRMBartTokenizer) -> bool:
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    prep_idx = tokenizer.convert_tokens_to_ids(PREP_PREFIX)

    for idx in reversed(range(len(input_ids))):  # Iterate in reverse to always get longer subsequences
        # If this token is :prep
        if input_ids[idx] == prep_idx:
            if len(input_ids[idx:]) == 1:  # incomplete :prep- (not followed by anything)
                return False
            else:  # first token of subsequence is :prep- and there are other tokens
                # we end in a valid :prep- if there are no spaces in this string, e.g. :prep-against
                decoded = tokenizer.decode_and_fix(input_ids[idx:])[0]
                return " " not in decoded and decoded.count(":") == 1

    return False


if __name__ == "__main__":
    tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")

    inputs = tokenizer(":startrel evidence :sense1 :prep-on-behalf-of :ref2454", add_special_tokens=False).input_ids

    print(ends_in_ref(inputs, tokenizer))
