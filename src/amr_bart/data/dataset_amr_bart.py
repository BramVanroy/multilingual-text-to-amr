import logging

from torch.utils.data import Dataset

from .IO import read_raw_amr_data


def collate_amr(tokenizer, samples):
    batch_sentences = [s["sentences"] for s in samples]
    encoded, extra = tokenizer.batch_encode_sentences(batch_sentences)
    extra["ids"] = [s["id"] for s in samples]
    encoded = {**encoded, **extra}

    if "linearized_graphs_ids" in samples[0]:
        batch_linearized_graphs = [s["linearized_graphs_ids"] for s in samples]
        encoded_graphs, extra_y = tokenizer.batch_encode_graphs_from_linearized(batch_linearized_graphs, samples)
        encoded = {**encoded, **encoded_graphs, **extra_y}

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

            try:
                self.tokenizer.batch_encode_sentences([graph.metadata["snt"]])
            except Exception:
                logging.warning("Invalid sentence when trying to tokenize it!")
                continue

            if remove_longer_than and len(linearized_graph) > remove_longer_than:
                continue
            if len(linearized_graph) > 1024:
                logging.warning("Sequence longer than 1024 included. BART does not support it!")

            self.sentences.append(graph.metadata["snt"])
            self.graphs.append(graph)
            self.linearized.append(linearized_graph)
            self.linearized_extra.append(linearized_extras)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {"id": idx, "sentences": self.sentences[idx]}
        sample = {**sample, **self.tokenizer(sample["sentences"])}
        if self.linearized is not None:
            sample["linearized_graphs_ids"] = self.linearized[idx]
            sample = {**sample, **self.linearized_extra[idx]}

        return sample
