"""Translate all the sentences in "# ::snt " in given AMR files with an M2M or NLLB translation model.
For the language keys, make sure to use a valid one for the corresponding model!

    For M2M: https://huggingface.co/facebook/m2m100_418M#languages-covered;
    For NLLB: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
"""
from collections import Counter
from dataclasses import field, dataclass
from os import PathLike
from pathlib import Path
from typing import Union, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def batch_sentences(sentences, batch_size: int = 32):
    batch_len = len(sentences)
    for idx in range(0, batch_len, batch_size):
        yield sentences[idx: min(idx + batch_size, batch_len)]


@dataclass
class Translator:
    model_name_or_path: str
    src_lang: str
    tgt_lang: str
    max_length: int = 256
    no_cuda: bool = False
    num_threads: int = None
    model: PreTrainedModel = field(default=None, init=False, repr=False)
    tokenizer: PreTrainedTokenizer = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not torch.cuda.is_available():
            self.no_cuda = True

        if self.no_cuda and self.num_threads:
            torch.set_num_threads(self.num_threads)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        if not self.no_cuda:
            self.model = self.model.to("cuda")

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

    def translate(self, sentences: List[str]):
        encoded = self.tokenizer(sentences, return_tensors="pt", padding=True)
        if not self.no_cuda:
            encoded = encoded.to("cuda")

        try:  # M2M
            generated_tokens = self.model.generate(
                **encoded, forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang), max_length=self.max_length
            )
        except AttributeError:
            generated_tokens = self.model.generate(
                **encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang], max_length=self.max_length
            )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate(
        amr_dir: Union[str, PathLike],
        output_dir: Union[str, PathLike],
        model_name_or_path: Union[str, PathLike] = "facebook/m2m100_418M",
        src_lang: str = "en",
        tgt_lang: str = "nl",
        batch_size: int = 32,
        max_length: int = 256,
        num_threads: int = None,
        no_cuda: bool = False,
        verbose: bool = False,
):
    """Given a directory of AMR, all .txt files will recursively be traversed and translated. All the lines
    that start with "# ::snt " will be translated. For the language keys, make sure to use a valid one!

    For M2M: https://huggingface.co/facebook/m2m100_418M#languages-covered
    For NLLB: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

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

    translator = Translator(model_name_or_path, src_lang, tgt_lang, max_length, no_cuda, num_threads)
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
                translations = translator.translate(list(b_sentences))

                for transl_idx, transl in zip(b_idxs, translations):
                    transl = " ".join(transl.split())
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
    cparser.add_argument(
        "-m", "--model_name_or_path", default="facebook/m2m100_418M", help="name or path of M2M translation model"
    )
    cparser.add_argument("--src_lang", default="en", help="language code of source language")
    cparser.add_argument("--tgt_lang", default="nl", help="language code of target language")
    cparser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=32,
        help="batch size of translating simultaneously. Set to lower value if you get out of memory issues",
    )
    cparser.add_argument("--max_length", type=int, default=256, help="max length to generate translations")
    cparser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="if not using CUDA, how many threads to use in torch for parallel operations."
             " By default, as many threads as available will be used",
    )
    cparser.add_argument("--no_cuda", action="store_true", help="whether to disable CUDA if it is available")
    cparser.add_argument("-v", "--verbose", action="store_true", help="whether to print translations to stdout")

    cargs = cparser.parse_args()
    translate(**vars(cargs))


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0 python src/amr_bart/utils/amr_translate.py data/amr_annotation_3.0/data/amrs/ data/amr_annotation_3.0/data/amrs_nl -m facebook/m2m100_1.2B --src_lang en --tgt_lang nl --batch_size 8
"""
