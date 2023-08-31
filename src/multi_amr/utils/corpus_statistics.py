import json
from os import PathLike
from pathlib import Path

from datasets import DatasetDict
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from tqdm import tqdm


TOKENIZER_NAMES = (
    "bigscience/bloomz-560m",
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-50-many-to-one-mmt",
    "google/mt5-base",
    "t5-base",
    "facebook/nllb-200-3.3B",
    "google/flan-t5-base",
)


def main(din: str | PathLike):
    pdin = Path(din)
    dsdict = DatasetDict.load_from_disk(pdin)
    stats = {tok: {k: {} for k in dsdict} for tok in TOKENIZER_NAMES}

    for tokenizer_name in tqdm(TOKENIZER_NAMES, unit="tokenizer"):
        tok_wrapper = AMRTokenizerWrapper.from_pretrained(tokenizer_name)
        max_length = tok_wrapper.tokenizer.model_max_length

        if max_length > 2048:  # Some max lengths are set to LARGE INT
            if tok_wrapper.tokenizer_type == TokenizerType.BLOOM:
                # Does not have a model_max_length specified (well actually it is set to LARGE INT)
                max_length = 2048
            elif tok_wrapper.tokenizer_type == TokenizerType.MBART:
                # mbart-large-cc25 is set to 1024 but mbart-large-50-may-to-one-mmt is set to LARGE INT
                max_length = 1024
            elif tok_wrapper.tokenizer_type == TokenizerType.T5:
                # mbart-large-cc25 is set to 1024 but mbart-large-50-may-to-one-mmt is set to LARGE INT
                # 1024 according to the mT5 paper
                max_length = 1024

        stats[tokenizer_name]["max_length"] = max_length
        stats["dataset"] = {}
        for split, dataset in tqdm(dsdict.items(), leave=False, unit="split"):
            if split not in stats["dataset"]:
                sent_lens = sorted([len(s.split()) for s in dataset["sentence"]])
                stats["dataset"][split] = {
                    "num_samples": len(dataset),
                    "max_num_ws_tokens": max(sent_lens),
                    "num_ws_tokens": sent_lens,
                }

            enc_sents = tok_wrapper(dataset["sentence"], return_tensors="pt", padding=True).input_ids
            # Max length
            stats[tokenizer_name][split]["max_subwordtok_len_sents"] = enc_sents.size(1)
            enc_sents_lens = sorted(
                [len([idx for idx in enc if idx != tok_wrapper.tokenizer.pad_token_id]) for enc in enc_sents]
            )
            # All lens
            stats[tokenizer_name][split]["subwordtok_lens_sents"] = enc_sents_lens
            # Num of seq that are larger than max length
            stats[tokenizer_name][split]["num_sent_gt_maxlength"] = sum(
                [1 for leng in enc_sents_lens if leng > max_length]
            )

            enc_lins = tok_wrapper(dataset["linearized_penman"], return_tensors="pt", padding=True).input_ids
            stats[tokenizer_name][split]["max_subwordtok_len_labels"] = enc_lins.size(1)
            enc_lins_lens = sorted(
                [len([idx for idx in enc if idx != tok_wrapper.tokenizer.pad_token_id]) for enc in enc_lins]
            )
            stats[tokenizer_name][split]["subwordtok_lens_labels"] = enc_lins_lens
            stats[tokenizer_name][split]["num_lbl_gt_maxlength"] = sum(
                [1 for leng in enc_lins_lens if leng > max_length]
            )

    pdin.joinpath("corpus_statistics.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Get some statistics for a number of tokenizers for a given preprocessed dataset"
    )
    cparser.add_argument(
        "din",
        help="directory containing the processed dataset. Output statistics will be written to a JSON file"
             " 'corpus_statistics.json' in this directory",
    )
    cargs = cparser.parse_args()
    main(cargs.din)
