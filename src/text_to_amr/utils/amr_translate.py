"""Translate all the sentences in "# ::snt " in given AMR files with an M2M translation model."""
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import Union

from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch


def batch_sentences(sentences, batch_size: int = 32):
    batch_len = len(sentences)
    for idx in range(0, batch_len, batch_size):
        yield sentences[idx:min(idx + batch_size, batch_len)]


def translate(amr_dir: Union[str, PathLike], output_dir: Union[str, PathLike],
              model_name_or_path: Union[str, PathLike] = "facebook/m2m100_418M",
              src_lang="en", tgt_lang="nl", batch_size=32, max_length=256, num_threads: int = None,
              no_cuda: bool = False, verbose: bool = False):
    """Given a directory of AMR, all .txt files will recursively be traversed and translated. All the lines
    that start with "# ::snt " will be translated

    :param amr_dir: dir with AMR files (potentially deep-structured)
    :param output_dir: dir to write the new structure and files in with translated sentences
    :param model_name_or_path: name or path of M2M translation model
    :param src_lang: language code of source language
    :param tgt_lang: language code of target language
    :param batch_size: batch size of translating simultaneously. Set to lower value if you get out of memory issues
    :param max_length: max length to generate translations
    :param num_threads: if not using CUDA, how many threads to use in torch for parallel operations. By default, as
    many threads as available will be used
    :param no_cuda: whether to disable CUDA if it is available
    :param verbose: whether to print translations to stdout
    """
    if not torch.cuda.is_available():
        no_cuda = True

    if no_cuda and num_threads:
        torch.set_num_threads(num_threads)

    model = M2M100ForConditionalGeneration.from_pretrained(model_name_or_path)

    if not no_cuda:
        model = model.to("cuda")

    model.eval()

    tokenizer = M2M100Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    pdin = Path(amr_dir).resolve()
    pdout = Path(output_dir).resolve()
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        pfout = pdout / pfin.relative_to(pdin)
        pfout.parent.mkdir(exist_ok=True, parents=True)
        with pfout.open("w", encoding="utf-8") as fhout:
            lines = pfin.read_text(encoding="utf-8").splitlines()
            # get sentece lines and remove "# ::snt " prefix
            sentences = [(line_idx, line[8:]) for line_idx, line in enumerate(lines) if line.startswith("# ::snt ")]

            for batch in tqdm(list(batch_sentences(sentences, batch_size=batch_size)), unit="batch", leave=False):
                b_idxs, b_sentences = zip(*batch)
                encoded = tokenizer(b_sentences, return_tensors="pt", padding=True)
                if not no_cuda:
                    encoded = encoded.to("cuda")
                generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                                                  max_length=max_length)
                translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                for transl_idx, transl in zip(b_idxs, translations):
                    if verbose:
                        print(f"ORIG: {lines[transl_idx][8:]}")
                        print(f"TRANSLATE: {transl}")
                        print()
                    lines[transl_idx] = f"# ::snt {transl}"

                fhout.write("\n".join(lines) + "\n")


def check_sequence_length(amr_dir: Union[str, PathLike]):
    """Naively check the number of tokens in the AMR files to get an idea of the sequence lengths.
    Utility function to call on new corpora if necessary. Not actively used.

    :param amr_dir: directory where AMR files are stored. All .txt files will be checked recursively
    """
    pdin = Path(amr_dir).resolve()
    clengths = Counter()
    for pfin in pdin.rglob("*.txt"):
        lines = pfin.read_text(encoding="utf-8").splitlines()
        sentences = [line for line in lines if line.startswith("# ::snt ")]
        sent_lengths = [len(sent.split(" ")) for sent in sentences]
        clengths.update(sent_lengths)

    clengths = {k: clengths[k] for k in sorted(clengths, reverse=True)}

    print("No. tokens of sentences, sorted high to low:")
    print(list(clengths.keys()))

    print("No. tokens of sentences and times that a sentence with such no. tokens occurs, sorted high to low:")
    print(clengths)


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("amr_dir", help="dir with AMR files (potentially deep-structured)")
    cparser.add_argument("output_dir", help="dir to write the new structure and files in with translated sentences")
    cparser.add_argument("--model_name_or_path", default="facebook/m2m100_418M",
                         help="name or path of M2M translation model")
    cparser.add_argument("--src_lang", default="en", help="language code of source language")
    cparser.add_argument("--tgt_lang", default="nl", help="language code of target language")
    cparser.add_argument("--batch_size", type=int, default=32,
                         help="batch size of translating simultaneously. Set to lower value if you get"
                              " out of memory issues")
    cparser.add_argument("--max_length", type=int, default=256, help="max length to generate translations")
    cparser.add_argument("--num_threads", type=int, default=None,
                         help="if not using CUDA, how many threads to use in torch for parallel operations."
                              " By default, as many threads as available will be used")
    cparser.add_argument("--no_cuda", action="store_true", help="whether to disable CUDA if it is available")
    cparser.add_argument("-v", "--verbose", action="store_true", help="whether to print translations to stdout")

    cargs = cparser.parse_args()
    translate(**vars(cargs))



if __name__ == "__main__":
    main()
