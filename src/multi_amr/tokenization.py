import copy
import logging
import sys


if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

from enum import auto
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import penman
import regex as re
import torch
from multi_amr.data.additional_tokens import AMR_TOKEN, get_added_vocabulary
from multi_amr.data.postprocessing_graph import (
    BACKOFF,
    ParsedStatus,
    connect_graph_if_not_connected,
    fix_and_make_graph,
)
from multi_amr.data.postprocessing_str import postprocess_str_after_delinearization, tokenize_except_quotes_and_angles
from multi_amr.utils import remove_wiki_from_graph
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
        tokens_to_add = get_added_vocabulary()
        tokens_to_add = set(tokens_to_add)
        voc = set(self.tokenizer.get_vocab().keys())
        new_tokens = list(sorted(tokens_to_add - voc))
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} new tokens to tok_wrapper")

        self.lang_idxs = None
        self.amr_token = AMR_TOKEN
        self.amr_token_id = self.tokenizer.convert_tokens_to_ids(self.amr_token)
        if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast)):
            self.tokenizer_type = TokenizerType.MBART
            self.tokenizer.tgt_lang = self.amr_token
            self.token_prefix = "\u2581"
            self.lang_idxs = set(self.tokenizer.lang_code_to_id.values())
        elif isinstance(self.tokenizer, (BartTokenizer, BartTokenizerFast)):
            self.tokenizer_type = TokenizerType.BART
            self.token_prefix = "\u0120"
        elif isinstance(self.tokenizer, (NllbTokenizer, NllbTokenizerFast)):
            self.tokenizer_type = TokenizerType.NLLB
            self.tokenizer.tgt_lang = self.amr_token
            self.token_prefix = "\u2581"
            self.lang_idxs = set(self.tokenizer.lang_code_to_id.values())
        elif isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            # T5 works with general prefixes that are part of the input, e.g. "Translate English to German: "
            # so there are no language codes
            self.tokenizer_type = TokenizerType.T5
            self.token_prefix = "\u2581"
        elif isinstance(self.tokenizer, BloomTokenizerFast):
            # BLOOM was not trained with prefixes. BLOOMZ was (also uses Bloomtokenizer) so there are no language codes
            # We will always use prefixes. There is no `slow` version
            self.tokenizer_type = TokenizerType.BLOOM
            self.token_prefix = "\u0120"
        else:
            raise ValueError(f"Tokenizer type '{type(self.tokenizer)}' not supported.")

        self.tokenizer.voc_size = len(self.tokenizer)

        self.vocab = self.tokenizer.get_vocab()
        self.added_vocab = self.tokenizer.get_added_vocab()
        assert self.amr_token in self.vocab
        assert self.amr_token_id in self.vocab.values()

        self.special_ids_and_amr_token_id = self.tokenizer.all_special_ids + [self.amr_token_id]

    @cached_property
    def num_spec_toks_when_building(self):
        input_ids = self.tokenizer.encode("This is not a potato.")
        input_ids_w_special = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return len(input_ids_w_special) - len(input_ids)

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
        max_length: Optional[int] = None,
    ) -> BatchEncoding:
        max_length = (max_length if max_length else self.tokenizer.model_max_length) - self.num_spec_toks_when_building
        all_token_ids = []

        self.tokenizer._switch_to_target_mode()
        for graph in graphs_or_penmanstrs:
            tokenized = self.tokenize_amr(graph, remove_wiki=remove_wiki, verbose=verbose)[:max_length]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
            token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
            # Replace language token with AMR token. We use this instead of tokenizer.as_target_tokenizer because
            # for some models it is hard to incorporate a new "real" language code token that will work with that
            if self.lang_idxs:
                token_ids = [self.amr_token_id if idx in self.lang_idxs else idx for idx in token_ids]
            all_token_ids.append(token_ids)

        self.tokenizer._switch_to_input_mode()

        # Create padded tensor
        max_seq_len = max([len(ids) for ids in all_token_ids])
        batch_size = len(all_token_ids)
        token_tensor = torch.full((batch_size, max_seq_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        for seq_idx, ids in enumerate(all_token_ids):
            token_tensor[seq_idx][: len(ids)] = torch.LongTensor(ids)

        return BatchEncoding(data={"input_ids": token_tensor}, tensor_type="pt")

    def tokenize_amr(
        self,
        graph_or_penman: penman.Graph | str,
        remove_wiki: bool = False,
        verbose: bool = False,
    ) -> List[str]:
        graph = graph_or_penman
        if isinstance(graph_or_penman, str):
            graph = penman.decode(graph_or_penman)
        if remove_wiki:
            graph = remove_wiki_from_graph(graph_or_penman)

        linearized_nodes = linearize(graph)
        if verbose:
            print("after linearization", linearized_nodes)

        bpe_tokens = []
        pref = self.token_prefix

        for tokk in linearized_nodes:
            # Special tokens are in vocab without prefix space but regular tokens might be with prefix
            is_in_voc = tokk in self.vocab
            is_prefixed_in_voc = (pref + tokk) in self.vocab
            is_rel = tokk.startswith(":") and len(tokk) > 1
            is_spc = tokk.startswith("<") and tokk.endswith(">")
            is_of = tokk.startswith(":") and tokk.endswith("-of")
            is_frame = re.match(r".+-\d\d$", tokk) is not None
            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace("_", " ")
                bpe_toks = ["<lit>"] + self.tokenizer.tokenize(tokk) + ["</lit>"]
            elif tokk == "(":
                bpe_toks = ["<rel>"]
            elif tokk == ")":
                bpe_toks = ["</rel>"]
            elif is_rel or is_spc or is_frame or is_of:
                if is_in_voc:
                    bpe_toks = [tokk]
                elif is_frame:
                    bpe_toks = self.tokenizer.tokenize(tokk[:-3]) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if rel in self.vocab:
                        bpe_toks = [rel, "-of"]
                    else:
                        bpe_toks = [":"] + self.tokenizer.tokenize(rel[1:]) + ["-of"]
                elif is_rel:
                    if "-" in tokk:
                        # E.g. ":prep-with" -> ":prep-", "with"
                        tokpref, rest = tokk.split("-", 1)
                        if tokpref + "-" in self.vocab:
                            bpe_toks = [tokpref + "-"] + self.tokenizer.tokenize(rest)
                        else:
                            bpe_toks = [":"] + self.tokenizer.tokenize(tokk[1:])
                    else:
                        bpe_toks = [":"] + self.tokenizer.tokenize(tokk[1:])
                else:
                    raise
            else:
                if is_prefixed_in_voc:
                    bpe_toks = [pref + tokk]
                elif is_in_voc:
                    bpe_toks = [tokk]
                else:
                    bpe_toks = self.tokenizer.tokenize(tokk)

            bpe_tokens.append(bpe_toks)

        bpe_tokens = [b for bb in bpe_tokens for b in bb]

        if verbose:
            print("after tokenization", bpe_tokens)
        return bpe_tokens

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

        if verbose:
            print("Raw ids2tokens (before remove special)", self.tokenizer.convert_ids_to_tokens(token_ids))

        if remove_special_tokens:
            token_ids = self.remove_special_tokens(token_ids)

        if verbose:
            print("Raw ids2tokens (after remove special)", self.tokenizer.convert_ids_to_tokens(token_ids))

        if not token_ids:
            return BACKOFF, ParsedStatus.BACKOFF, None

        try:
            decoded = self.tokenizer.decode(token_ids)
            if verbose:
                print("decoded with tokenizer", decoded)
            postprocessed = postprocess_str_after_delinearization(decoded)
            if verbose:
                print("after postprocess str", postprocessed)
            nodes = tokenize_except_quotes_and_angles(postprocessed)
            if verbose:
                print("After tokenization", nodes)
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
                    return graph, status, nodes
                except Exception as e:
                    if verbose:
                        print("Reconnection 2 failure:", file=sys.stderr)
                        print(e, file=sys.stderr)
                        print(nodes, file=sys.stderr)
                        print(graph_, file=sys.stderr)
                        print(token_ids, file=sys.stderr)
                    return BACKOFF, ParsedStatus.BACKOFF, nodes


def linearize(graph: penman.Graph) -> List[str]:
    # modified from SPRING
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}
    try:
        linearized = penman.encode(graph_).replace("–", "-")  # NLLB does not have an en-hyphen
    except Exception as exc:
        print(graph_)
        print(graph_.metadata)
        raise exc

    linearized_nodes = _tokenize_encoded_graph(linearized)
    remap = {}
    for i in range(1, len(linearized_nodes)):
        nxt = linearized_nodes[i]
        lst = linearized_nodes[i - 1]
        if nxt == "/":
            remap[lst] = f"<pointer:{len(remap)}>"

    i = 1
    linearized_nodes_ = [linearized_nodes[0]]
    while i < (len(linearized_nodes)):
        nxt = linearized_nodes[i]
        lst = linearized_nodes_[-1]
        if nxt in remap:
            if lst == "(" and linearized_nodes[i + 1] == "/":
                nxt = remap[nxt]
                i += 1
            elif lst.startswith(":"):
                nxt = remap[nxt]
        elif lst == ":polarity" and nxt == "-":
            linearized_nodes_[-1] = ":negation"
            i += 1
            continue
        linearized_nodes_.append(nxt)
        i += 1

    linearized_nodes_ = [tstrip for t in linearized_nodes_ if (tstrip := t.strip())]
    return linearized_nodes_


def _tokenize_encoded_graph(linearized: str) -> List[str]:
    # modified from SPRING
    linearized = re.sub(r"(\".+?\")", r" \1 ", linearized)
    pieces = []
    for piece in linearized.split():
        if piece.startswith('"') and piece.endswith('"'):
            pieces.append(piece)
        else:
            piece = piece.replace("(", " ( ")
            piece = piece.replace(")", " ) ")
            piece = piece.replace(":", " :")
            piece = piece.replace("/", " / ")
            piece = piece.strip()
            pieces.append(piece)
    linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()
    return linearized.split(" ")
