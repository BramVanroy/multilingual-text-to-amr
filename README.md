# multilingual-text-to-amr
 
An adaptation of MBART to parse text into AMR for multiple languages.

# Usage

First install the package by running the following in the root directory where `setup.py` is present.

```shell
pip install -e .
```

Then, training or evaluating is as simple as updating the configuration file `example_config` to specify where
your data is and any other parameters. Note that for every language you can specify a directory. All .txt files
in that directory will be recursively included in the dataset. Make sure to also specify which source languages
each dataset has! Specify those in `src_langs`.

After completing the config file, simply run the entry point on your config:

```shell
run-mbart-amr example_config.json
```

For development, it can be useful to limit the training/evaluation samples to have a quick run through the whole
pipeline. You can specify those with

```json
"max_train_samples_per_language": null,
"max_eval_samples_per_language": null,
```

I checked the input of the English AMR 3.0 corpus, and very few samples have a tokenized input (text) of more than
`128`. However, I recommend to not change the output length (set it to `null` or leave it out), because that will 
truncate the linearized tree (labels), which leads to an invalid reference tree!

These defaults should be sensible:

```json
"input_max_seq_length": 128,
"output_max_seq_length": null,
```

If you want to have a look at all the possible arguments, you can run:

```shell
run-mbart-amr -h
```

Out-of-the-box the code should be compatible with distributed systems although I have not tested this explicitly. The 
following should work for single node, multi-gpu systems. Where `--nproc_per_node` is the number of GPUs
and `OMP_NUM_THREADS` is the number of CPU threads divided by number of GPUs.

```shell
OMP_NUM_THREADS=6 python -m torch.distributed.run --standalone --nproc_per_node 2 src/mbart_amr/run_mbart_amr.py config.json
```

# Architecture and tokenizer
For now, the MBART architecture can be used as-is with the exception of added vocabulary items (i.e. increasing the embedding size a
little bit; currently 121 new tokens). These added vocabulary items can be found as a
[Python iterable](src/mbart_amr/data/tokens.py) in this repo. This also adds `amr_XX`, which is used
as the "special language code" for generating AMR. The description of all tokens that we add is given later in this
README.

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

## Added tokens

Sources:

- [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md)
- [AMR dictionary](https://amr.isi.edu/doc/amr-dict.html)

### Relations and roles

The list of added tokens is mostly inspired by the [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#part-ii--concepts-and-relations).

For `:ARGX` we add up to `:ARG10`. For `:opX` up to `:op9`. This does not mean that higher ranks, such as `:op25`, 
are not possible. But it does mean that they are not based on a single token but need to be generated. Therefore,
the model can still generate outliers like `:op25` with multiple subtokens, i.e. `:op2`
and `5`. The same is true for `:ARGX`, tokens.

### Entity types (not added)

AMR describes specific entity types in [the guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#named-entities).
However, because these seem semantically already related to "real" words, we do not explicitly add these as whole
tokens in hopes that the language model already has a good "understanding" of these tokens.

### Special frames for roles

Some special frames are possible in AMR. These always end with `-91`, e.g. `have-degree-91` (for comparatives and
superlatives). To make these roles more generic, we only add `-91` as a token so the model can learn to generate them.
This format is the same as for senses (which end in e.g. `-01` for sense 01). So technically, a sense `-91`
can be ambiguous with a special frame that ends in `-91`. We will assume that `-91` always indicates a special frame's 
role and never a sense because a sense 91 is probably quite rare.

It can also occur as part of, e.g., a phone number. Unfortunately we have no solution for that ambiguity so when
tokenizing there might be inconsistencies in how spaces are handled after the token as `-91` will now always have
a space after it but no space before it.

### Quantity

AMR offers a wide range of [quantity specifiers](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#quantities).
These specifiers always end with `-quantity`, so to make this more generic, we only add `-quantity` as a token.

### Other entities

Some [other entities](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#other-entities-dates-times-percentages-phone-email-urls)
are also allowed, such as `date-entity` or `phone-number-entity`. To make this more generic, we only add `-entity`.

### Special cases

- In exceptions, when no better annotation is found, prepositions can also be annotated by `:prep-` + the prefix, e.g.,
`prep-by`. Therefore, we also add `prep-` to the vocabulary. 
- Similarly, not all conjunctions are covered in AMR so they can be generically created with `:conj-`, which
we add to the vocabulary.
- All relations have an inverse relation that is created with the `-of` suffix, e.g., `:ARG0-of`. However, because `-of`
is such a frequent token, it would be too ambiguous in the data. Instead we add `~~of` and account for that in the
linearization process.
- Questions in AMR are denoted with the special relation `amr-unknown`, which we add. (It is therefore similar to `?`)
- Choice questions are a bit different, they are marked with `amr-choice`, which we add. (Similar to `or`)
- Negative polarity `:polarity -` is a way of negating what is being said. This is so frequent that we add a special
`:negation` token. (Similar to `not`)

### Sentences

AMR allows multi-sentences constructions (which can also be phrases). Sentences are denoted with `:sntX`. We add
up to `:snt9`. Because it occurs often, we also add the special token `multi-sentence`.

### Senses

AMR tracks different senses of words with OntoNotes specifiers, e.g. `charge-05`. We use `:senseX` instead.

Note the difference with other numbers arguments that we add. Here, the sense ID is the same as the sense ID in the 
corpus, meaning that it counts `00`, `01`, `02` etc, and not `0` `1` `2`. Therefore, we also add `:sense1` in addition
`:sense01` so that the model can also generated `:sense12` automatically, based on `:sense1` + `2`.

### Reference tokens and their IDs

In AMR, we often refer back to other concepts by means of variables. We us `:refX` as variables and add up to 9 
(e.g. `:ref9`). Higher numbers can be generated dynamically.

### Branch boundaries

Because we are predicting a linearized version of AMR, we need a way to encapsulate branches, i.e., to indicate when
a branch starts and stops. In AMR, these branches are typically relations so we do not need an explicit start relation
tag but we do indicate the end of a branch with `:endrel`.

### Subsets

[Subsets](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#subsets) are covered with `:subset`, which we add.
Due to the generic token `-of` that is in the vocabulary, `:subset-of` can automatically be generated.

### Tree start/end

We will assume that the start and end of the prediction indicate the start and end of the tree. So no special tokens
are needed there. However, MBART relies on special language tokens to learn language-specific phenomena (and even
translate). Therefore, we add the special token `amr_XX` which we will use as a special token.


## TODO

- postprocessing for invalid trees that the model may produce;
- add script to only run prediction on an already trained model so that we can go from text -> AMR. Must include postprocessing!
- add conditional decoding in beam search (if time)
- add hyperparameter search (maybe)


## Verify with partners

- Is it a problem that we use duplication (over languages) in training/eval?

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
