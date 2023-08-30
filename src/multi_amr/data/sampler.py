import logging
from collections import defaultdict
from typing import Iterator, List

import torch
from datasets import Dataset
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


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
    src_langs = [sample["src_lang_idx"] for sample in dataset]
    is_predict = len([d["linearized_penman"] for d in dataset if d["linearized_penman"]]) == 0

    if is_predict:
        logger.warning(
            "Detected 'predict' mode, so switching defaults to: group_by_length=False, "
            " keep_incomplete_batches=True, shuffle=False"
        )
        group_by_length = False
        keep_incomplete_batches = True
        shuffle = False

    # Per language, collect all the indices assosicated with that language
    per_lang_idxs = defaultdict(list)
    for idx, src_lang in enumerate(src_langs):
        per_lang_idxs[src_lang].append(idx)
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
            lengthed = []

            def find_lengths(sample, idx):
                lengthed.append((idx, len(sample["sentence"]), len(sample["linearized_penman"])))
                return None

            dataset.map(find_lengths, with_indices=True)

            # Sort by input and output lengths, and extract lang_idxs
            lang_idxs = torch.LongTensor([tup[0] for tup in sorted(lengthed, key=lambda tup: tup[1:], reverse=True)])

        for batch in lang_idxs.split(batch_size, dim=0):
            if batch.size(0) == batch_size:
                lang_batches.append(batch)
            else:
                rest_batches.append(batch)

    # Do shuffle of full batches across languages
    if lang_batches:
        lang_batches: torch.Tensor = torch.stack(lang_batches)
        if shuffle:
            full_batch_indices = torch.randperm(lang_batches.size(0), generator=generator)
            lang_batches = lang_batches[full_batch_indices]
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


class DistributedSrcLangGroupedSampler(DistributedSampler):
    """Sampler that samples indices in a way that groups together source languages while also having some randomness"""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *args,
        keep_incomplete_batches: bool = True,
        group_by_length: bool = True,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)
        self.batch_size = batch_size
        self.dataset = dataset
        self.keep_incomplete_batches = keep_incomplete_batches
        self.group_by_length = group_by_length

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_src_lang_grouped_indices(
            self.dataset,
            self.batch_size,
            keep_incomplete_batches=self.keep_incomplete_batches,
            shuffle=self.shuffle,
            group_by_length=self.group_by_length,
            generator=g,
        )

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
