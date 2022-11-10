from collections import defaultdict
from typing import Iterator, List

import torch
from mbart_amr.data.dataset import AMRDataset
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


def get_src_lang_grouped_indices(
    src_langs: List[str],
    batch_size: int,
    keep_incomplete_batches: bool = False,
    shuffle: bool = True,
    generator=None,
):
    """Group the data by the source language, so that each batch consists the same language only. The final few batches
    may contain mixed languages, however, because they contain the data per language that did not fit in a single batch.
    By default, these are dropped. To include these incomplete batches, set 'keep_incomplete_batches = True'.

    :param src_langs: a list of strings, indicating the source language of each sample
    :param batch_size: the batch size to use, must be same as in the dataloader
    :param keep_incomplete_batches: whether to keep incomplete batches. This will cause the final batch(es) to have data
    of different languages!
    :param shuffle: whether to shuffle the indices per language and then shuffle the batches across all languages as
    well as the rest category language batch(es). In the rest category that means that all data points per-language are
    kept together but that the order of the languages can shuffle.
    :param generator: optional torch generator
    :return: indices of the dataset, ordered in such a way so that they are homogenous in their source language (with
    the potential exception of the last batch(es))
    """
    # Per language, collect all the indices assosicated with that language
    lang_idxs = defaultdict(list)
    for idx, src_lang in enumerate(src_langs):
        lang_idxs[src_lang].append(idx)

    lang_batches = []
    rest_batches = []
    # Iterate over a language and all its associated indices so that we can
    # 1. shuffle the data (could have done it outside the loop as well for ALL at once, but here we are)
    # 2. split the indices into batches. We keep track of full batches and incomplete ones so that we can
    # 3. put all the incomplete batches near the end or drop them in case keep_incomplete_batches = False
    for lang, lang_idxs in lang_idxs.items():
        lang_idxs = torch.LongTensor(lang_idxs)
        if shuffle:
            # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
            # Do shuffle within a language
            indices = torch.randperm(lang_idxs.size(0), generator=generator)
            lang_idxs = lang_idxs[indices]

        for batch in lang_idxs.split(batch_size, dim=0):
            if batch.size(0) == batch_size:
                lang_batches.append(batch)
            else:
                rest_batches.append(batch)

    # Do shuffle of full batches across languages
    if lang_batches:
        lang_batches = torch.stack(lang_batches)
        if shuffle:
            full_batch_indices = torch.randperm(lang_batches.size(0), generator=generator)
            lang_batches = lang_batches[full_batch_indices]
        lang_batches = lang_batches.flatten().tolist()

    if not lang_batches and not keep_incomplete_batches:
        raise ValueError(
            "Your total batch_size (batch_size * accumulation_steps) is larger than your number of samples per language."
            " This means we can only generate incomplete batches. But because 'keep_incomplete_batches' is False, those"
            " are ignored. As a result, no data will be used. Try lowering the total batch size, or add more data."
            " (Check that you have not set max_train_samples to a low value!)"
        )

    if keep_incomplete_batches and rest_batches:
        rest_batches = torch.stack(rest_batches)
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
        batch_size: int,
        dataset: AMRDataset,
        keep_incomplete_batches: bool = False,
        shuffle: bool = True,
        generator=None,
    ):
        self.batch_size = batch_size
        self.src_langs = [d["metadata"]["src_lang"] for d in dataset]
        self.keep_incomplete_batches = keep_incomplete_batches
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        return len(self.src_langs)

    def __iter__(self) -> Iterator:
        indices = get_src_lang_grouped_indices(
            self.src_langs,
            self.batch_size,
            self.keep_incomplete_batches,
            shuffle=self.shuffle,
            generator=self.generator,
        )
        return iter(indices)


class DistributedSrcLangGroupedSampler(DistributedSampler):
    """Sampler that samples indices in a way that groups together source languages while also having some randomness"""

    def __init__(self, batch_size: int, dataset: AMRDataset, *args, keep_incomplete_batches: bool = True, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.batch_size = batch_size
        self.src_langs = [d["metadata"]["src_lang"] for d in dataset]
        self.keep_incomplete_batches = keep_incomplete_batches

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_src_lang_grouped_indices(
            self.src_langs, self.batch_size, self.keep_incomplete_batches, shuffle=self.shuffle, generator=g
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
