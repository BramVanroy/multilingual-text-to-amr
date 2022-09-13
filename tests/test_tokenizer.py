from amr_mbart.amr_mbart.tokenization_amr_bart import AMRBartTokenizer
from amr_mbart.data.dataset_amr_mbart import AMRDataset, AMRDatasetTokenBatcherAndLoader


def main():
    tokenizer = AMRBartTokenizer.from_pretrained("facebook/bart-large")
    ds = AMRDataset([r"D:\corpora\amr_annotation_3.0\data\amrs\split\test\amr-release-3.0-amrs-test-consensus.txt"],
                    tokenizer)
    dl = AMRDatasetTokenBatcherAndLoader(ds)

    for batch in dl:
        print(len(batch))
        for item in batch:
            print(item)
        exit()

if __name__ == '__main__':
    main()