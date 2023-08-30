import logging
import sys
from collections import Counter
from typing import List, Optional

import torch
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def collate_amr(
    samples: List[dict],
    src_langs: List[str],
    tok_wrapper: AMRTokenizerWrapper,
    input_max_seq_length: Optional[int] = None,
    output_max_seq_length: Optional[int] = None,
):
    """Collate a given batch from the dataset by 1. tokenizing a given sentence and getting its attention mask,
    token_ids, etc. for input; 2. linearizing and tokenizing the associated penman str as the labels.

    :param samples: a given batch
    :param src_langs: a list of languages, can be 'English' or 'en_XX' depending on the model
    :param tok_wrapper: modified AMR tokenizer wrapper to use
    :param input_max_seq_length: optional max sequence length to truncate the input data to
    :param output_max_seq_length: optional max sequence length to truncate the output data (labels) to
    :return: a dictionary with keys such as token_ids and labels, with values tensors
    """
    model_max_length = tok_wrapper.tokenizer.model_max_length
    if input_max_seq_length is not None and input_max_seq_length > model_max_length:
        raise ValueError(
            f"'input_max_seq_length' ({input_max_seq_length}) cannot be larger than max model size ({model_max_length})"
        )
    if output_max_seq_length is not None and output_max_seq_length > model_max_length:
        raise ValueError(
            f"'output_max_seq_length' ({output_max_seq_length}) cannot be larger than max model size ({model_max_length})"
        )

    src_lang_idxs = Counter([s["src_lang_idx"] for s in samples])
    src_lang_idx = src_lang_idxs.most_common(1)[0][0]
    src_lang = src_langs[src_lang_idx]
    if len(src_lang_idxs.keys()) > 1:
        logging.warning(
            "This given batch consists of multiple source language. Therefore, the tok_wrapper will"
            f" append a single language code ({src_lang}) that is not applicable to all samples, which may"
            f" lead to poor performance."
        )

    task_prefix = ""
    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB):
        # Set the source lang to the main language in this batch so that the correct token can be added (not used by T5)
        tok_wrapper.tokenizer.src_lang = src_lang
    elif tok_wrapper.tokenizer_type in (TokenizerType.T5, TokenizerType.BLOOM):
        # T5 can use prefixes. Lower case "translate", like in T5 pretraining
        task_prefix = f"translate {src_lang} to {tok_wrapper.amr_token}: "

    has_labels = bool(len([s["linearized_penman"] for s in samples if s["linearized_penman"]]))
    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.T5):
        # ENCODER-DECODERS
        batch = tok_wrapper(
            [task_prefix + s["sentence"] for s in samples],
            padding=True,
            truncation=True,
            max_length=input_max_seq_length,
            return_tensors="pt",
        )

        if has_labels:
            batch["labels"] = tok_wrapper(
                [s["linearized_penman"] for s in samples],
                max_length=output_max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            batch["labels"] = torch.where(batch["labels"] == tok_wrapper.tokenizer.pad_token_id, -100, batch["labels"])
    else:
        if input_max_seq_length is None or output_max_seq_length is None:
            max_length = None
        else:
            max_length = input_max_seq_length + output_max_seq_length
            if max_length > model_max_length:
                raise ValueError(
                    f"'input_max_seq_length' + 'output_max_seq_length' ({max_length}) cannot be larger"
                    f" than max model size ({model_max_length}) for decoder-only models"
                )

        batch = tok_wrapper(
            [
                task_prefix
                + s["sentence"]
                + "\n"
                + tok_wrapper.tokenizer.eos_token
                + (
                    s["linearized_penman"] if has_labels else ""
                )  # Only add "labels" in train/eval mode, not in predict
                for s in samples
            ],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if has_labels:
            # Labels are a copy of the token_ids (causal LM). They are shifted within the forward pass so we do not
            # have to do that here. We do set the prompt (prefix + sentence + "\n" + EOS) to -100 to ignore it in loss.
            batch["labels"] = batch["token_ids"].clone()
            labels = batch["labels"]
            eos_token_id = tok_wrapper.tokenizer.eos_token_id
            idxs_to_remove = []
            for sample_idx in range(labels.size(0)):
                # We find the first index where EOS occurs. Everything before it (and itself) is ignored
                # It is possible that we do not find an index when the max length is shorter than the input sentence
                # because EOS is placed after the input sentence in CLM (and after that the linearized AMR)
                try:
                    end_of_prompt = (labels[sample_idx] == eos_token_id).nonzero(as_tuple=True)[0][0]
                except IndexError:
                    idxs_to_remove.append(sample_idx)
                else:
                    labels[sample_idx].index_fill_(0, torch.arange(start=0, end=end_of_prompt + 1), -100)

            if idxs_to_remove:
                logging.warning(
                    f"Removed {len(idxs_to_remove):,} samples because they were so long that no EOS was found."
                )
                # Only keep the items that have EOS
                mask = torch.ones_like(batch["labels"], dtype=torch.bool)
                mask[idxs_to_remove] = False
                batch["labels"] = torch.masked_select(batch["labels"], mask)

    return batch
