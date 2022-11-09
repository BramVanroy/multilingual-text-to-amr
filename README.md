# multilingual-text-to-amr
 
An adaptation of MBART to parse text into AMR for multiple languages.

# Usage

First install the package by running the following in the root directory where `setup.py` is present.

```shell
pip install .
```

Then, training or evaluating is as simple as updating the configuration file `example_config` to specify where
your data is and any other parameters. Note that for every language you can specify a directory. All .txt files
in that directory will be recursively included in the dataset. Make sure to also specify which source languages
each dataset has! Specify those in `src_langs`.

After completing the config file, simply run the entry point:

```shell
python src/mbart_amr/run_mbart_amr.py example_config.json
```

For development, it can be useful to limit the training/evaluation samples to have a quick run through the whole
pipeline. You can specify those with

```json
"max_train_samples_per_language": null,
"max_eval_samples_per_language": null,
```

If you want to have a look at all the possible arguments, you can run:

```shell
python src/mbart_amr/run_mbart_amr.py -h
```

Out-of-the-box the code should be compatible with distributed systems although I have not tested this explicitly.

# Architecture and tokenizer
For now, the MBART architecture can be used as-is with the exception of added vocabulary items (i.e. increasing the embedding size a
little bit; currently 122 new tokens). These added vocabulary items can be found as a
[text file](src/mbart_amr/data/vocab/additions.txt) in this repo. This also adds `amr_XX`, which is used
as the "special language code" for generating AMR. The description of all tokens that we add is given in this
[README](src/mbart_amr/data/vocab/README.md).

The [tokenizer](src/mbart_amr/data/tokenization.py) is updated to add AMR-specific functionality:

- encoding penman AMR strings to token IDs by linearizing and then tokenizing with `.encode_penmanstrs()`;
- decoding tokenized and linearized AMRs with `.decode_amr_and_fix()`. Note that the `clean_up_tokenization()` function
is paramount to make sure that everything is working well to solve spacing issues left by the tokenizer.

This approach was tested on the whole AMR 3.0 corpus. It can successfully go from all trees, to a linearized 
version, to a tokenized version, back to a linearized version, and back to the same original tree. See 
[this test](tests/test_tokenization.py).

# Linearization

[linearization.py](src/mbart_amr/data/linearization.py)

I decided to linearize AMR by starting from the penman `Tree` (not the graph). The reason being that graphs can be 
cyclical, which I did not want to deal with. By recursively iterating over the tree and making use of the annotation
schema for pattern matching, I linearize a tree in a deterministic way in `penmantree2linearized()`. Most importantly,
to be able to go back from linearized to a tree, I needed a way to add "depth" to the flat string. So `:startrel` and 
`:endrel` mark starts and ends of branches in the tree.

A linearized tree can be turned back into a tree by `linearized2penmantree()`. Recursion on a flat string is hard but
because of the `:startrel` and `:endrel` tokens, we are still capable of correctly putting a tree together again.

This approach was tested on the whole AMR 3.0 corpus. It can successfully go from all trees, to a linearized 
version, and back to the same original tree. See [this test](tests/test_linearization.py).


# Caveats on using the linearizer and tokenizer with the AMR corpus

There are some annotation quirks in the AMR corpus that may need to be dealt with implicitly. Similarly, the MBART
tokenizer may output characters that are not encoded in the same way as the input (that's why we add `fix_text`) in the 
tokenizer's decode method. When running through a dataset, you should take the following steps for encoding:

1. run the penman string to ftfy's `fix_text` because of encoding issues in the AMR corpus;
2. run `tree.reset_variables()` because in annotation these can be rather random
(see [this issue](https://github.com/goodmami/penman/issues/112));
3. go back to a penman string with `penman.format()`;
4. run `fix_text` over the formatted tree;
5. run the tokenizer's `.encode_penmanstrs()` method on the penman string. For training, we probably do not want to
include wiki entries, so we can remove those with the argument `remove_wiki=True`.

Here is an example

```python
import penman
from ftfy import fix_text
from src.mbart_amr.data.tokenization import AMRMBartTokenizer
from src.mbart_amr.data.linearization import linearized2penmanstr

penman_str = "(d / dog :ARG0-of (b / bark-01))"  # A penman string representing the AMR graph
# NOTE: the fix_text is important to make sure the reference tree also is correctly formed, e.g.
# (':op2', '"d’Intervention"'), -> (':op2', '"d\'Intervention"'),
penman_str = fix_text(penman_str)

tree = penman.parse(penman_str)

tree.reset_variables()
penman_str = penman.format(tree)

tokenizer = AMRMBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
encoded = tokenizer.encode_penmanstrs(penman_str, remove_wiki=True)
decoded = tokenizer.decode_amr_and_fix(encoded.input_ids)
decoded_penman = linearized2penmanstr(decoded)
```

## AMR guidelines

https://github.com/amrisi/amr-guidelines/blob/master/amr.md

## TODO

- postprocessing for invalid trees that the model may produce;
- add smatch as metric;
- add script to only run prediction on an already trained model so that we can go from text -> AMR. Must include postprocessing!
- smart initialization of our special tokens by copying weights from the existing embedding to the new tokens;
- add conditional decoding in beam search (if time)
- add hyperparameter search (maybe)

## Verify with partners

- Is it a problem that we use duplication (over languages) in training/eval?

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
