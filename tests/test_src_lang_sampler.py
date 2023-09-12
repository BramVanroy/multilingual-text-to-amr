from collections import Counter

from datasets import DatasetDict

from multi_amr.data.sampler import get_src_lang_grouped_indices


def main(preprocessed_data: str):
    batch_size = 8
    ddataset = DatasetDict.load_from_disk(preprocessed_data)

    for split_name, dataset in ddataset.items():
        for group_by_length in (True, False):
            for shuffle in (True, False):
                for keep_incomplete_batches in (True, False):
                    batch_idxs = get_src_lang_grouped_indices(dataset=dataset, batch_size=8, group_by_length=group_by_length, shuffle=shuffle, keep_incomplete_batches=keep_incomplete_batches)

                    for idx in range(0, len(dataset)-batch_size, batch_size):
                        idxs = batch_idxs[idx:idx+batch_size]
                        batch = dataset.select(idxs)
                        counts = Counter([b["src_lang_idx"] for b in batch])
                        if len(counts) > 1:
                            print("Not homogeneous!",
                                  idxs,
                                  f"group_by_length: {group_by_length},"
                                  f" shuffle: {shuffle},"
                                  f" keep_incomplete_batches: { keep_incomplete_batches},"
                                  f" split_name: {split_name}")



if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Brute-force testing of AMR tokenization by tokenizing a linearized "
        "tree, tokenizing it, decoding it, and reconstructing the tree."
        " Then checking if the original and reconstructed trees are equal."
        " This script is a naive, brute-force way of testing this. All .txt"
        " files in a given directory will be (recursively) tested."
    )
    cparser.add_argument(
        "preprocessed_dataset",
        help="Directory to a preprocessed dataset dict that has been saved_to_disk.",
    )

    cargs = cparser.parse_args()
    main(cargs.preprocessed_dataset)
