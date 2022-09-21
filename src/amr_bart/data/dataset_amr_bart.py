import logging
import random
from functools import cached_property

import torch
from torch.utils.data import Dataset

from ..amr_bart.tokenization_amr_bart import AMRBartTokenizer
from .IO import read_raw_amr_data


def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y["decoder_input_ids"], y["lm_labels"][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x["input_ids"][:, :-1]
    lm_labels = x["input_ids"][:, 1:]
    x = {"input_ids": input_ids, "attention_mask": attention_mask}
    y = {"decoder_input_ids": decoder_input_ids, "lm_labels": lm_labels}
    return x, y


def collate_amr(tokenizer: AMRBartTokenizer, samples):
    batch_sentences = [s["sentences"] for s in samples]
    encoded, extra = tokenizer.batch_encode_sentences(batch_sentences)
    extra["ids"] = [s["id"] for s in samples]
    encoded = {**encoded, **extra}

    if "linearized_graphs_ids" in samples[0]:
        batch_linearized_graphs = [s["linearized_graphs_ids"] for s in samples]
        encoded_graphs, extra_y = tokenizer.batch_encode_graphs_from_linearized(batch_linearized_graphs, samples)
        encoded = {**encoded, **extra_y}

    return encoded


class AMRDataset(Dataset):
    def __init__(
        self,
        paths,
        tokenizer,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than

        for graph in read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify):
            linearized_graph, linearized_extras = self.tokenizer.linearize(graph)

            # try:
            #     self.tokenizer.batch_encode_sentences([g.metadata["snt"]])
            # except Exception:
            #     logging.warning("Invalid sentence when trying to tokenize it!")
            #     continue
            #
            # if remove_longer_than and len(l) > remove_longer_than:
            #     continue
            # if len(l) > 1024:
            #     logging.warning("Sequence longer than 1024 included. BART does not support it!")

            self.sentences.append(graph.metadata["snt"])
            self.graphs.append(graph)
            self.linearized.append(linearized_graph)
            self.linearized_extra.append(linearized_extras)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {"id": idx, "sentences": self.sentences[idx]}

        if self.linearized is not None:
            sample["linearized_graphs_ids"] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])

        return sample

    @staticmethod
    def size(sample):
        return len(sample["linearized_graphs_ids"])


class AMRDatasetTokenBatcherAndLoader:
    """TODO: (BV) rework for distributed mode"""

    def __init__(self, dataset, batch_size=800, shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = [[self.dataset[s] for s in b] for b in it]
        it = (collate_amr(self.tokenizer, b) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]

        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

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
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
