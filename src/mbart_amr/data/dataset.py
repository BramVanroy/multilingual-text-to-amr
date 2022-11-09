from os import PathLike
from pathlib import Path
from typing import Union, List, Optional

from ftfy import fix_text
import penman
from torch.utils.data import Dataset
from tqdm import tqdm

from mbart_amr.data.linearization import do_remove_wiki
from mbart_amr.data.tokenization import AMRMBartTokenizer

KEEP_KEYS = {
    "input_ids",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "head_mask",
    "decoder_head_mask",
    "cross_attn_head_mask",
    "encoder_outputs",
    "past_key_values",
    "inputs_embeds",
    "decoder_inputs_embeds",
    "labels",
}


def collate_amr(samples: List[dict],
                tokenizer: AMRMBartTokenizer,
                input_max_seq_length: Optional[int] = None,
                output_max_seq_length: Optional[int] = None,
                ):
    """Collate a given batch from the dataset by 1. tokenizing a given sentence and getting its attention mask,
    token_ids, etc. for input; 2. linearizing and tokenizing the associated penman str as the labels.

    :param tokenizer: modified AMR tokenizer to use
    :param input_max_seq_length: optional max sequence length to truncate the input data to
    :param output_max_seq_length: optional max sequence length to truncate the output data (labels) to
    :param samples: a given batch
    :return: a dictionary with keys such as input_ids and labels, with values tensors
    """
    encoded_inputs = tokenizer([s["sentence"] for s in samples],
                               padding=True,
                               truncation=True,
                               max_length=input_max_seq_length,
                               return_tensors="pt")
    encoded_linearized = {"labels": tokenizer.encode_penmanstrs([s["penmanstr"] for s in samples],
                                                                padding=True,
                                                                truncation=True,
                                                                max_length=output_max_seq_length,
                                                                return_tensors="pt").input_ids}

    return {**encoded_inputs, **encoded_linearized}


class AMRDataset(Dataset):
    def __init__(
            self,
            din: Union[str, PathLike],
            remove_wiki: bool = False,
            max_samples: Optional[int] = None
    ):
        self.pdin = Path(din)
        self.remove_wiki = remove_wiki
        self.max_samples = max_samples

        self.sentences = []
        self.penmanstrs = []
        self.metadatas = []

        n_samples = 0
        for pfin in tqdm(list(self.pdin.rglob("*.txt")), unit="file"):
            with pfin.open(encoding="utf-8") as fhin:
                for tree in penman.iterparse(fhin):
                    tree.reset_variables()
                    # NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
                    # (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
                    penman_str = fix_text(penman.format(tree))
                    if self.remove_wiki:
                        penman_str = do_remove_wiki(penman_str)
                    self.sentences.append(tree.metadata["snt"])
                    self.penmanstrs.append(penman_str)
                    self.metadatas.append(tree.metadata)

                    n_samples += 1

                    if self.max_samples and n_samples == self.max_samples:
                        break

            if self.max_samples and n_samples == self.max_samples:
                break

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return {"id": idx,
                "sentence": self.sentences[idx],
                "penmanstr": self.penmanstrs[idx],
                "metadata": self.metadatas[idx],
                }
