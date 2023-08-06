import logging
from enum import StrEnum, auto
from typing import Dict, List

import numpy as np
import penman
import torch
from multi_amr.data.additional_tokens import get_added_vocabulary
from multi_amr.data.postprocessing_graph import tokens2graph
from multi_amr.data.postprocessing_str import (
    clean_up_amr_tokenization,
    postprocess_str_after_delinearization,
    postprocess_str_after_linearization,
)
from multi_amr.data.prepare_dataset import dfs_linearize, remove_wiki_from_graph
from transformers import (
    AutoTokenizer,
    BloomTokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    PreTrainedTokenizerBase,
    T5Tokenizer,
    T5TokenizerFast,
)


logger = logging.getLogger(__name__)


class TokenizerType(StrEnum):
    BLOOM = auto()
    MBART = auto()
    NLLB = auto()
    T5 = auto()


class AMRTokenizerWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

        self.amr_token = "<AMR>"
        if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            self.tokenizer_type = TokenizerType.MBART
            self.tokenizer.tgt_lang = self.amr_token  # AMR is always target in our case
            self.lang_idxs = torch.LongTensor(list(self.tokenizer.lang_code_to_id.values()))
            self.token_prefix = "\u2581"
        elif isinstance(self.tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            self.tokenizer_type = TokenizerType.NLLB
            self.tokenizer.tgt_lang = self.amr_token  # AMR is always target in our case
            self.lang_idxs = torch.LongTensor(list(self.tokenizer.lang_code_to_id.values()))
            self.token_prefix = "\u2581"
        elif isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            # T5 works with general prefixes that are part of the input, e.g. "Translate English to German: "
            # so there are no language codes
            self.tokenizer_type = TokenizerType.T5
            self.lang_idxs = None
            self.token_prefix = "\u2581"
        elif isinstance(self.tokenizer, BloomTokenizerFast):
            # BLOOM was not trained with prefixes. BLOOMZ was (also uses Bloomtokenizer) so there are no language codes
            # We will always use prefixes. There is no `slow` version
            self.tokenizer_type = TokenizerType.BLOOM
            self.lang_idxs = None
            self.token_prefix = "\u0120"
        else:
            raise ValueError(f"Tokenizer type '{type(self.tokenizer)}' not supported.")

        tokens_to_add = get_added_vocabulary(prefix=self.token_prefix)
        tokens_to_add = set(tokens_to_add)
        voc = set(self.tokenizer.get_vocab().keys())
        new_tokens = list(sorted(tokens_to_add - voc))

        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new tokens to tok_wrapper")

        # Just adding AMR to voc and defining it here as the tgt_lang is not enough
        # because we are always just calling the tok_wrapper (as if it were the source tok_wrapper)
        # However, we cannot even use it as a target tok_wrapper with tgt_lang AMR, because
        # the MBARTTokenizer only allows special language codes as tgt_lang for this purpose so
        # we cannot take that approach. Instead we will be replacing the special source language
        # token in "encode_penmanstrs" with our own, AMR one
        self.tokenizer.voc_size = len(self.tokenizer)

        self.added_vocab = self.tokenizer.get_added_vocab()
        self.amr_token_idx = self.tokenizer.convert_tokens_to_ids("<AMR>")
        self.all_special_ids_tensor = torch.LongTensor(self.tokenizer.all_special_ids + [self.amr_token_idx])

    @classmethod
    def from_pretrained(cls, *args, legacy: bool = True, **kwargs):
        """The legacy option is important because a recent fix was submitted to fix T5 tokenization but this seems to
         have the opposite effect here, leading to `unks`. So we keep the legacy behavior for now.
         TODO: monitor the status of this PR (https://github.com/huggingface/transformers/pull/25224)
             and pin version of transformers in pyproject if this PR is merged in new release (likely 4.31.1 or 4.32.0)

        See https://github.com/huggingface/transformers/pull/24565
        """
        return cls(tokenizer=AutoTokenizer.from_pretrained(*args, legacy=legacy, **kwargs))

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def batch_encode_penmanstrs(
        self, penmanstrs: List[str], remove_wiki: bool = False, remove_metadata: bool = True, **tokenizer_kwargs
    ):
        if isinstance(penmanstrs, str):
            raise TypeError("Input 'penmanstrs' must be a list of strings")

        linearizeds = []
        for penmanstr in penmanstrs:
            graph = penman.decode(penmanstr)
            if remove_metadata:
                graph.metadata = []

            if remove_wiki:
                graph = remove_wiki_from_graph(graph)

            linearized = " ".join(dfs_linearize(graph))
            linearized = postprocess_str_after_linearization(linearized)
            linearizeds.append(linearized)

        return self.tokenizer(linearizeds, **tokenizer_kwargs)

    def batch_decode_amr_ids(self, all_token_ids: List[List[int]], verbose:bool = False, **tokenizer_kwargs) -> Dict[str, List]:
        """
        Returns a dict with `penman` and `status` keys.
        """
        if isinstance(all_token_ids, torch.Tensor):
            if all_token_ids.dim() == 1:
                token_ids = all_token_ids.unsqueeze(dim=0)
        elif isinstance(all_token_ids, np.ndarray):
            if all_token_ids.ndim == 1:
                token_ids = np.expand_dims(all_token_ids, axis=0)
        elif isinstance(all_token_ids[0], int):
            token_ids = [all_token_ids]

        if not isinstance(all_token_ids, torch.Tensor):
            all_token_ids = torch.LongTensor(all_token_ids)

        output = {"penman": [], "status": []}
        for token_ids in all_token_ids:
            token_ids = self.remove_special_tokens(token_ids)
            print("after remove special", token_ids)
            decoded = self.tokenizer.decode(token_ids, **tokenizer_kwargs)
            print(decoded)
            sequence = clean_up_amr_tokenization(decoded)
            print(sequence)
            sequence = postprocess_str_after_delinearization(sequence)
            print(sequence)
            graph, status = tokens2graph(sequence.split(), verbose=verbose)
            output["penman"].append(penman.encode(graph))
            output["status"].append(status)

        return output

    def remove_special_tokens(self, input_ids: torch.LongTensor):
        """NOTE: only removes special tokens and AMR, NOT the added tokens.
        Does not work on batches (because not every sequences is equally long as required by torch tensors)"""

        # Because `<AMR>` is not a real "special token", it does not get ignored so we have to remove it ourselves
        # It is included in all_special_ids_tensor
        return input_ids[~torch.isin(input_ids, self.all_special_ids_tensor)]

