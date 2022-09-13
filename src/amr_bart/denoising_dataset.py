import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@contextlib.contextmanager
def fixed_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


@dataclass
class DenoisingDataset(Dataset):
    """Adapted for MBART. Won't work well for BART because the input format is different.
    - In MBART, the format is `<tokens><EOS><LID>` where `tokens` can contain multiple sentences
    separated by EOS
    - In BART, the format is `<BOS><tokens><EOS>`

    because there is no special token at the start in MBART, preprocessing is a bit different (we include index 0 in
     most operations)
    """

    tokenizer: PreTrainedTokenizer
    dataset: HfDataset
    permute_sentence_ratio: float = 1.0
    mask_ratio: float = 0.3
    random_ratio: float = 0.1
    insert_ratio: float = 0.0
    rotate_ratio: float = 0.0
    poisson_lambda: float = 3.5
    replace_length: int = 1
    mask_length: Optional[str] = "span-poisson"
    mask_whole_word: Optional[torch.LongTensor] = None
    seed: Optional[int] = None
    full_stop_index: Optional[int] = None

    def __post_init__(self):
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}. Has to be {{-1, 0, 1}}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(
                f"invalid arg: mask-length={self.mask_length}. Has to be {{'subword', 'word'," f" 'span-poisson'}}"
            )
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError("if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.full_stop_index = (
            self.full_stop_index if self.full_stop_index is not None else self.tokenizer.eos_token_id
        )

    def __getitem__(self, index):
        with fixed_seed(self.seed, index):
            input_ids: torch.LongTensor = self.dataset[index]["input_ids"]
            labels = input_ids.clone()

            assert isinstance(input_ids, torch.LongTensor)

            if self.permute_sentence_ratio > 0.0:
                input_ids = self.permute_sentences(input_ids, self.permute_sentence_ratio)
                print("done permute")

            if self.mask_ratio > 0:
                input_ids = self.add_whole_word_mask(input_ids, self.mask_ratio)
                print("done mask")

            if self.insert_ratio > 0:
                input_ids = self.add_insertion_noise(input_ids, self.insert_ratio)
                print("done insert")

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                input_ids = self.add_rolling_noise(input_ids)
                print("done rotate")

        assert (input_ids >= 0).all()
        assert (input_ids[1:-1] >= 1).all()
        assert (input_ids <= len(self.tokenizer)).all()

        input_ids = input_ids

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.dataset)

    def permute_sentences(self, source, p=1.0):
        full_stops = source == self.full_stop_index
        # Pretend it ends with a full stop so last span is a sentence

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        index = 0
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 0) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            index += sentence.size(0)

        return result

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero()
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.tokenizer.mask_token_id
            source[indices[mask_random]] = torch.randint(1, len(self.tokenizer), size=(mask_random.sum(),))

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.tokenizer.mask_token_id
                    source[indices[mask_random]] = torch.randint(1, len(self.tokenizer), size=(mask_random.sum(),))
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.tokenizer.mask_token_id
                    source[indices[mask_random]] = torch.randint(1, len(self.tokenizer), size=(mask_random.sum(),))

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.tokenizer.mask_token_id
        result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.tokenizer), size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result
