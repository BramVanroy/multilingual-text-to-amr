import logging
from enum import StrEnum, auto
from typing import Dict, List, Tuple

import penman
import torch
from multi_amr.data.additional_tokens import get_added_vocabulary
from multi_amr.data.linearization import dfs_linearize, remove_wiki_from_graph
from multi_amr.data.postprocessing_graph import ParsedStatus, tokens2graph
from multi_amr.data.postprocessing_str import (
    postprocess_str_after_delinearization,
    postprocess_str_after_linearization,
    tokenize_except_quotes_and_angles,
)
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

        # tokens_to_add = get_added_vocabulary(prefix=self.token_prefix)
        tokens_to_add = get_added_vocabulary()
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
        self.special_ids_and_amr_token_id = self.tokenizer.all_special_ids + [self.amr_token_idx]

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
        self,
        penmanstrs: List[str],
        remove_wiki: bool = False,
        remove_metadata: bool = True,
        verbose: bool = False,
        **tokenizer_kwargs,
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
            linearized = postprocess_str_after_linearization(linearized, verbose=verbose)
            linearizeds.append(linearized)

        return self.tokenizer(linearizeds, **tokenizer_kwargs)

    def batch_decode_amr_ids(
        self, all_token_ids: List[List[int]], verbose: bool = False, reset_variables: bool = False, **tokenizer_kwargs
    ) -> Dict[str, List]:
        """
        Returns a dict with `penman` and `status` keys.
        """
        output = {"penman": [], "status": []}
        for token_ids in all_token_ids:
            penmanstr, status = self.decode_amr_ids(
                token_ids, verbose=verbose, reset_variables=reset_variables, **tokenizer_kwargs
            )
            output["penman"].append(penmanstr)
            output["status"].append(status)

        return output

    def decode_amr_ids(
        self,
        token_ids: List[int],
        verbose: bool = False,
        reset_variables: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **tokenizer_kwargs,
    ) -> Tuple[str, ParsedStatus]:
        token_ids = self.remove_special_tokens(token_ids)
        decoded = self.tokenizer.decode(
            token_ids, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **tokenizer_kwargs
        )
        if verbose:
            print("after decoding", decoded)

        sequence = postprocess_str_after_delinearization(decoded)
        if verbose:
            print("after postprocess", sequence)
        graph, status = tokens2graph(tokenize_except_quotes_and_angles(sequence), verbose=verbose)

        if reset_variables:
            # To tree so that we can reset the variables
            tree = penman.configure(graph)
            try:
                tree.reset_variables()
            except Exception as exc:
                print(sequence)
                print(graph)
                raise exc
            penmanstr = penman.format(tree)
        else:
            penmanstr = penman.encode(graph)

        return penmanstr, status

    def remove_special_tokens(self, input_ids: List[int]):
        """NOTE: only removes special tokens and AMR, NOT the added tokens.
        Does not work on batches (because not every sequences is equally long as required by torch tensors)"""

        # Because `<AMR>` is not a real "special token", it does not get ignored so we have to remove it ourselves
        # It is included in special_ids_and_amr_token_id
        return [idx for idx in input_ids if idx not in self.special_ids_and_amr_token_id]
