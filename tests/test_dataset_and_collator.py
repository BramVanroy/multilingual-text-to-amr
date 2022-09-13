import torch
from datasets import Dataset
from transformers import MBartTokenizer

from amr_mbart.denoising_collator import DenoisingCollator
from amr_mbart.denoising_dataset import DenoisingDataset


def test_dataset_and_collator():
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX")
    text = ["UN Chief Says There Is No Military Solution in Syria",
            f"I have never seen this man before in my life{tokenizer.eos_token}But I like him though!{tokenizer.eos_token}Maybe we can meet tomorrow or any other day this week?!"]
    input_ids = [e for e in tokenizer(text)["input_ids"]]
    ds = Dataset.from_dict({"input_ids": input_ids})
    ds.set_format("torch")

    denoise_ds = DenoisingDataset(tokenizer, ds, seed=42)

    batch = [denoise_ds[0], denoise_ds[1]]
    collate_fn = DenoisingCollator(tokenizer)
    batch = collate_fn(batch)

    print("text", text)
    print("input_ids", tokenizer.batch_decode(batch["input_ids"]))
    print("labels", tokenizer.batch_decode([[l for l in seq if l != -100] for seq in batch["labels"]]))
    print("decoder_input_ids", tokenizer.batch_decode(batch["decoder_input_ids"]))

    assert torch.equal(batch["input_ids"], torch.LongTensor([[8274, 250026, 2071, 438, 67485, 53, 187895, 23, 51712,
                                                              2, 250004, 1, 1, 1, 1, 1, 1, 1,
                                                              1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                              1, 1, 1, 1, 1],
                                                             [87, 765, 8306, 46280, 903, 250026, 2, 83425, 642,
                                                              831, 23356, 127773, 707, 2499, 3789, 5155, 903, 5895,
                                                              4730, 2, 4966, 139222, 21208, 38, 2, 250004, 1,
                                                              1, 1, 1, 1, 1]]))

    assert torch.equal(batch["labels"], torch.LongTensor([[8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53,
                                                           187895, 23, 51712, 2, 250004, -100, -100, -100, -100,
                                                           -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                                           -100, -100, -100, -100, -100],
                                                          [87, 765, 8306, 51592, 903, 332, 8108, 23, 759,
                                                           6897, 2, 4966, 87, 1884, 4049, 21208, 38, 2,
                                                           83425, 642, 831, 23356, 127773, 707, 2499, 3789, 5155,
                                                           903, 5895, 4730, 2, 250004]]))

    assert torch.equal(batch["decoder_input_ids"],
                       torch.LongTensor([[250004, 8274, 127873, 25916, 7, 8622, 2071, 438, 67485,
                                          53, 187895, 23, 51712, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1],
                                         [250004, 87, 765, 8306, 51592, 903, 332, 8108, 23,
                                          759, 6897, 2, 4966, 87, 1884, 4049, 21208, 38,
                                          2, 83425, 642, 831, 23356, 127773, 707, 2499, 3789,
                                          5155, 903, 5895, 4730, 2]]))

    assert torch.equal(batch["attention_mask"],
                       torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 0, 0, 0, 0, 0, 0]]))

    assert torch.equal(batch["decoder_attention_mask"],
                       torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1]]))


if __name__ == "__main__":
    test_dataset_and_collator()
