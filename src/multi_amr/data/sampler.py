import logging
import random
from collections import defaultdict
from statistics import mean
from typing import Iterator, List

import torch
from datasets import Dataset
from multi_amr.data.tokenization import AMRTokenizerWrapper
from torch.utils.data import Sampler


logger = logging.getLogger(__name__)


def get_src_lang_grouped_indices(
    dataset: Dataset,
    batch_size: int,
    keep_incomplete_batches: bool = False,
    shuffle: bool = True,
    group_by_length: bool = True,
    generator=None,
):
    """Group the data by the source language, so that each batch consists the same language only. The final few batches
    may contain mixed languages, however, because they contain the data per language that did not fit in a single batch.
    By default, these are dropped. To include these incomplete batches, set 'keep_incomplete_batches = True'.

    :param dataset: dataset to process
    :param batch_size: the batch size to use, must be same as in the dataloader
    :param keep_incomplete_batches: whether to keep incomplete batches. This will cause the final batch(es) to have data
    of different languages!
    :param shuffle: whether to shuffle the indices per language and then shuffle the batches across all languages as
    well as the rest category language batch(es). In the rest category that means that all data points per-language are
    kept together but that the order of the languages can shuffle.
    :param group_by_length: whether to try and group batches by similar input and output lengths
    :param generator: optional torch generator
    :return: indices of the dataset, ordered in such a way so that they are homogenous in their source language (with
    the potential exception of the last batch(es))
    """
    all_src_lang_idxs = [sample["src_lang_idx"] for sample in dataset]
    is_predict = len([d["penmanstr"] for d in dataset if "penmanstr" in d and d["penmanstr"]]) == 0

    if is_predict:
        logger.warning(
            "Detected 'predict' mode, so switching defaults to: group_by_length=False, "
            " keep_incomplete_batches=True, shuffle=False"
        )
        group_by_length = False
        keep_incomplete_batches = True
        shuffle = False

    sampleidx2lengths = {}

    def find_lengths(sample, sample_idx):
        if is_predict:
            sampleidx2lengths[sample_idx] = (len(sample["sentence"]),)
        else:
            sampleidx2lengths[sample_idx] = (len(sample["sentence"]), len(sample["penmanstr"]))
        return None

    dataset.map(find_lengths, with_indices=True)

    # Per language, collect all the indices assosicated with that language
    per_lang_idxs = defaultdict(list)
    for sample_idx, src_lang_idx in enumerate(all_src_lang_idxs):
        per_lang_idxs[src_lang_idx].append(sample_idx)
    per_lang_idxs = dict(per_lang_idxs)
    lang_batches: List = []
    rest_batches: List = []
    # Iterate over a language and all its associated indices so that we can
    # 1. shuffle the data (could have done it outside the loop as well for ALL at once, but here we are)
    # 2. split the indices into batches. We keep track of full batches and incomplete ones so that we can
    # put all the incomplete batches near the end or drop them in case keep_incomplete_batches = False
    for lang, lang_idxs in per_lang_idxs.items():
        lang_idxs = torch.LongTensor(lang_idxs)
        if shuffle:
            # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
            # Do shuffle within a language
            indices = torch.randperm(lang_idxs.size(0), generator=generator)
            lang_idxs = lang_idxs[indices]

        if group_by_length:
            # Sort by input and output lengths. Largest first
            lang_idxs = torch.LongTensor(
                sorted(lang_idxs.tolist(), key=lambda lang_idx: sampleidx2lengths[lang_idx][1:], reverse=True)
            )

        for batch in lang_idxs.split(batch_size, dim=0):
            if batch.size(0) == batch_size:
                lang_batches.append(batch)
            else:
                rest_batches.append(batch)

    def sort_batches_by_avg_length(batches: torch.Tensor) -> torch.LongTensor:
        batches = batches.tolist()
        # Only averages by input length here (not output)
        avg_lens = [
            (batch_idx, mean([sampleidx2lengths[idx][0] for idx in _batch]))
            for batch_idx, _batch in enumerate(batches)
        ]
        batches = torch.LongTensor([batches[tup[0]] for tup in sorted(avg_lens, key=lambda tup: tup[1], reverse=True)])
        return batches

    # Do shuffle of full batches across languages
    if lang_batches:
        lang_batches: torch.Tensor = torch.stack(lang_batches)
        if shuffle:
            full_batch_indices = torch.randperm(lang_batches.size(0), generator=generator)
            lang_batches = lang_batches[full_batch_indices]
            # Re-sort to make sure that, even with a bit of randomness from the shuffle, the batches themselves
            # are also sorted by length
            lang_batches = sort_batches_by_avg_length(lang_batches)
        lang_batches = lang_batches.flatten().tolist()

    if not lang_batches and not keep_incomplete_batches:
        raise ValueError(
            "Your total batch_size (batch_size * accumulation_steps) is larger than your number of samples per language."
            " This means we can only generate incomplete batches. But because 'keep_incomplete_batches' is False, those"
            " are ignored. As a result, no data will be used. Try lowering the total batch size, add more data, or"
            " - if doing predictions - set keep_incomplete_batches to True."
        )

    if keep_incomplete_batches and rest_batches:
        rest_batches: torch.Tensor = torch.stack(rest_batches)
        # Do shuffle of incomplete batches across languages
        if shuffle:
            incomplete_batch_indices = torch.randperm(rest_batches.size(0), generator=generator)
            rest_batches = rest_batches[incomplete_batch_indices]
            rest_batches = sort_batches_by_avg_length(rest_batches)
        rest_batches = rest_batches.flatten().tolist()
        return lang_batches + rest_batches
    else:
        return lang_batches


class SrcLangGroupedSampler(Sampler):
    """Sampler that samples indices in a way that groups together source languages while also having some randomness"""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        keep_incomplete_batches: bool = False,
        shuffle: bool = True,
        group_by_length: bool = True,
        generator=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.keep_incomplete_batches = keep_incomplete_batches
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.generator = generator
        super().__init__(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator:
        indices = get_src_lang_grouped_indices(
            self.dataset,
            self.batch_size,
            self.keep_incomplete_batches,
            shuffle=self.shuffle,
            group_by_length=self.group_by_length,
            generator=self.generator,
        )
        return iter(indices)


class SpringSampler:
    def __init__(self, dataset, tok_wrapper: AMRTokenizerWrapper, batch_size_tokens: int = 500, shuffle: bool = False):
        self.dataset = dataset
        # TODO change back to shuffle
        self.shuffle = False
        self.tok_wrapper = tok_wrapper
        self.batch_size_tokens = batch_size_tokens
        # Listify so that we can get __len__ when needed
        self.batches = list(self._prepare_batches())

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _prepare_batches(self):
        # Modified from SPRING
        ids = list(range(len(self.dataset)))[::-1]

        if self.shuffle:
            random.shuffle(ids)

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            seq_length = self.tok_wrapper.batch_encode_amr([self.dataset[idx]["penmanstr"]]).input_ids.size(1)
            cand_batch_ntokens = max(seq_length, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size_tokens and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, seq_length)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size_tokens:
                yield discharge()

        if batch_ids:
            yield discharge()
