import copy
import logging
import sys
from enum import StrEnum, auto
from typing import Dict, List, Optional, Tuple

import penman
import regex as re
import torch
from multi_amr.data.additional_tokens import get_added_vocabulary
from multi_amr.data.linearization import dfs_linearize, remove_wiki_from_graph
from multi_amr.data.postprocessing_graph import (
    BACKOFF,
    ParsedStatus,
    connect_graph_if_not_connected,
    fix_and_make_graph,
    token_processing,
)
from multi_amr.data.postprocessing_str import (
    postprocess_str_after_delinearization,
    postprocess_str_after_linearization,
    tokenize_except_quotes_and_angles,
)
from transformers import (
    AutoTokenizer,
    BartTokenizer,
    BartTokenizerFast,
    BatchEncoding,
    BloomTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
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
    BART = auto()
    NLLB = auto()
    T5 = auto()


class AMRTokenizerWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.patterns = re.compile(
            r" ?<[a-z]+:?\d*>| ?:\S+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )

        if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast)):
            self.tokenizer_type = TokenizerType.MBART
            self.token_prefix = "\u2581"
            self.amr_token = f"{self.token_prefix}<AMR>"
            self.tokenizer.tgt_lang = self.amr_token  # AMR is always target in our case
            self.lang_idxs = torch.LongTensor(list(self.tokenizer.lang_code_to_id.values()))
        elif isinstance(self.tokenizer, (BartTokenizer, BartTokenizerFast)):
            self.tokenizer_type = TokenizerType.BART
            self.lang_idxs = None
            self.token_prefix = "\u0120"
        elif isinstance(self.tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            self.tokenizer_type = TokenizerType.NLLB
            self.token_prefix = "\u2581"
            self.amr_token = f"{self.token_prefix}<AMR>"
            self.tokenizer.tgt_lang = self.amr_token  # AMR is always target in our case
            self.lang_idxs = torch.LongTensor(list(self.tokenizer.lang_code_to_id.values()))
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

        self.amr_token = "<AMR>"
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

        self.vocab = self.tokenizer.get_vocab()
        self.added_vocab = self.tokenizer.get_added_vocab()
        self.amr_token_idx = self.tokenizer.convert_tokens_to_ids(self.amr_token)
        assert self.amr_token in self.vocab

        self.special_ids_and_amr_token_id = self.tokenizer.all_special_ids + [self.amr_token_idx]

    @classmethod
    def from_pretrained(cls, *args, legacy: bool = True, add_prefix_space: bool = True, **kwargs):
        """The legacy option is important because a recent fix was submitted to fix T5 tokenization but this seems to
         have the opposite effect here, leading to `unks`. So we keep the legacy behavior for now.
         TODO: monitor the status of this PR (https://github.com/huggingface/transformers/pull/25224)
             and pin version of transformers in pyproject if this PR is merged in new release (likely 4.31.1 or 4.32.0)

        See https://github.com/huggingface/transformers/pull/24565
        """
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(*args, legacy=legacy, add_prefix_space=add_prefix_space, **kwargs)
        )

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def remove_special_tokens(self, input_ids: List[int]) -> List[int]:
        """NOTE: only removes special tokens and AMR, NOT the added tokens.
        Does not work on batches (because not every sequences is equally long as required by torch tensors)"""

        # Because `<AMR>` is not a real "special token", it does not get ignored so we have to remove it ourselves
        # It is included in special_ids_and_amr_token_id
        # >0 ensures that ignored positions (like -100) are also removed
        return [idx for idx in input_ids if idx not in self.special_ids_and_amr_token_id and idx >= 0]

    def batch_encode_amr(
        self,
        graphs_or_penmanstrs: List[str | penman.Graph],
        remove_wiki: bool = False,
        verbose: bool = False,
        **tokenizer_kwargs,
    ) -> BatchEncoding:
        linearizeds = []
        for graph in graphs_or_penmanstrs:
            if isinstance(graph, str):
                ## TODO: add model optoin (dereify)
                graph = penman.decode(graph)
            if remove_wiki:
                graph = remove_wiki_from_graph(graph)
            # Modified from SPRING
            linearized_nodes = dfs_linearize(graph)
            linearized = " ".join(linearized_nodes)
            linearizeds.append(postprocess_str_after_linearization(linearized, verbose=verbose))

        return self(linearizeds, **tokenizer_kwargs)

    def batch_decode_amr_ids(
        self,
        all_token_ids: List[List[int]],
        verbose: bool = False,
        remove_special_tokens: bool = True,
    ) -> Dict[str, List]:
        """
        Returns a dict with `penman` and `status` keys.
        """
        output = {"graph": [], "status": []}
        for token_ids in all_token_ids:
            graph, status, _ = self.decode_amr_ids(
                token_ids, verbose=verbose, remove_special_tokens=remove_special_tokens
            )
            output["graph"].append(graph)
            output["status"].append(status)

        return output

    def decode_amr_ids(
        self,
        token_ids: List[int] | torch.Tensor,
        verbose: bool = False,
        remove_special_tokens: bool = True,
    ) -> Tuple[penman.Graph, ParsedStatus, None | List[str]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if remove_special_tokens:
            token_ids = self.remove_special_tokens(token_ids)

        if not token_ids:
            return BACKOFF, ParsedStatus.BACKOFF, None

        try:
            decoded = self.tokenizer.decode(token_ids)
            if verbose:
                print("decoded before postprocess_str", decoded)
            postprocessed = postprocess_str_after_delinearization(decoded)
            nodes = tokenize_except_quotes_and_angles(postprocessed)
            if verbose:
                print("nodes after postprocess str", nodes)
            nodes_ = nodes
        except Exception as e:
            if verbose:
                print("Decoding failure:", file=sys.stderr)
                print(e, file=sys.stderr)
                print(token_ids, file=sys.stderr)
            return BACKOFF, ParsedStatus.BACKOFF, None
        else:
            try:
                graph_ = graph = fix_and_make_graph(nodes, verbose=verbose)
            except Exception as e:
                if verbose:
                    print("Building failure:", file=sys.stderr)
                    print(nodes, file=sys.stderr)
                    print(e, file=sys.stderr)
                    print(token_ids, file=sys.stderr)
                return BACKOFF, ParsedStatus.BACKOFF, None
            else:
                try:
                    graph, status = connect_graph_if_not_connected(graph)
                    if status == ParsedStatus.BACKOFF and verbose:
                        print("Reconnection 1 failure:")
                        print(nodes, file=sys.stderr)
                        print(graph_, file=sys.stderr)
                        print(token_ids, file=sys.stderr)
                    return graph, status, nodes_
                except Exception as e:
                    if verbose:
                        print("Reconnection 2 failure:", file=sys.stderr)
                        print(e, file=sys.stderr)
                        print(nodes, file=sys.stderr)
                        print(graph_, file=sys.stderr)
                        print(token_ids, file=sys.stderr)
                    return BACKOFF, ParsedStatus.BACKOFF, nodes_

    def decode_into_nodes(self, token_ids: List[int]) -> List[str]:
        # This is the original in SPRING but, bug? Shouldn't there be a ":"?
        rex_arg = re.compile(f"^{self.token_prefix}(op|snt|conj|prep)")
        rex_spc = re.compile(r"<(s|/s|lit|/lit|stop|unk|pad|mask)>")

        # get strings
        subtokens = [t for t in self.tokenizer.convert_ids_to_tokens(token_ids) if t != self.tokenizer.pad_token]

        # subword collapse
        tokens = []
        subword_to_token_map = {}
        current_token_i = 0
        for subw_i, subtok in enumerate(subtokens):
            subword_to_token_map[subw_i] = current_token_i

            # if empty you cannot do anything but add a new word
            if not tokens:
                tokens.append(subtok.lstrip(self.token_prefix))
                current_token_i += 1
            # after a special token release
            elif isinstance(tokens[-1], str) and rex_spc.match(tokens[-1]):
                tokens.append(subtok.lstrip(self.token_prefix))
                current_token_i += 1

            # after a subtoken ':' (which should be followed by the rest of the edge) ignore self.token_prefix
            # TODO: this is an ugly patch due to the fact that BART tokenizer splits after ':'
            elif (tokens[-1] == ":") and rex_arg.match(subtok):
                tokens[-1] = tokens[-1] + subtok[1:]

            # leading self.token_prefix
            elif subtok.startswith(self.token_prefix):
                tokens.append(subtok.lstrip(self.token_prefix))
                current_token_i += 1

            # very ugly patch for some cases in which self.token_prefix is not in the following token to the edge
            elif (
                isinstance(tokens[-1], str)
                and tokens[-1].startswith(":")
                and tokens[-1][-1].isdigit()
                and (subtok != "-of")
            ):
                tokens.append(subtok.lstrip(self.token_prefix))
                current_token_i += 1

            # in any other case attach to the previous
            else:
                tokens[-1] = tokens[-1] + subtok

        # strip INIT and fix byte-level
        tokens = [
            self.tokenizer.convert_tokens_to_string(list(t)).lstrip() if isinstance(t, str) else t for t in tokens
        ]
        # tokens = [t.replace(self.token_prefix, '') if isinstance(t, str) else t for t in tokens]

        # unks are substituted with thing
        tokens = [t if t != self.tokenizer.unk_token else "thing" for t in tokens]

        old_tokens = tokens

        # <lit> Barack Obama </lit> -> "Barack Obama"
        tokens = []
        token_to_token_map = {}
        start_search = 0
        removed = 0
        while True:
            try:
                lit_start = old_tokens.index("<lit>", start_search)
                token_addition = old_tokens[start_search:lit_start]
                for i, t in enumerate(token_addition, start=start_search):
                    token_to_token_map[i] = i - removed
                tokens += token_addition
                lit_end = min(lit_start + 2, len(old_tokens) - 1)

                while lit_end < len(old_tokens):
                    old_tok = old_tokens[lit_end]

                    if isinstance(old_tok, str) and (
                        (old_tok.startswith(":") and len(old_tok) > 3) or (old_tok == "<stop>")
                    ):
                        res_tok = old_tokens[lit_start + 1 : lit_end]
                        for i in range(lit_start, lit_end):
                            token_to_token_map[i] = len(tokens)

                        # Remove possible wrong None
                        res = old_tokens[lit_start + 1 : lit_end]
                        res = [str(r) for r in res if r is not None]
                        res = '"' + "_".join(res) + '"'

                        removed += len(res_tok)
                        start_search = lit_end
                        tokens += [res, old_tok]
                        break

                    elif old_tok == "</lit>":
                        res_tok = old_tokens[lit_start + 1 : lit_end]
                        for i in range(lit_start, lit_end + 1):
                            token_to_token_map[i] = len(tokens)

                        # Remove possible wrong None
                        res = old_tokens[lit_start + 1 : lit_end]
                        res = [str(r) for r in res if r is not None]
                        res = '"' + "_".join(res) + '"'

                        removed += len(res_tok) + 1
                        start_search = lit_end + 1
                        tokens.append(res)
                        break

                    else:
                        lit_end += 1
                        start_search = lit_end

            except ValueError:
                token_addition = old_tokens[start_search:]
                for i, t in enumerate(token_addition, start=start_search):
                    token_to_token_map[i] = i - removed

                tokens += token_addition
                break

        tokens = [token_processing(t) for t in tokens]

        return tokens
