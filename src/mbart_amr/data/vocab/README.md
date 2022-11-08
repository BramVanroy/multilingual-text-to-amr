# Added tokens

Sources:

- [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md)
- [AMR dictionary](https://amr.isi.edu/doc/amr-dict.html)

## Relations and roles

The list of added tokens is mostly inspired by the [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#part-ii--concepts-and-relations).

For `:ARGX` we add up to `:ARG10`. For `:opX` up to `:op9`. This does not mean that higher ranks, such as `:op25`, 
are not possible. But it does mean that they are not based on a single token but need to be generated. Therefore,
the model can still generate outliers like `:op25` with multiple subtokens, i.e. `:op2`
and `5`. The same is true for `:ARGX`, tokens.

## Entity types (not added)

AMR describes specific entity types in [the guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#named-entities).
However, because these seem semantically already related to "real" words, we do not explicitly add these as whole
tokens in hopes that the language model already has a good "understanding" of these tokens.

## Special frames for roles

Some special frames are possible in AMR. These always end with `-91`, e.g. `have-degree-91` (for comparatives and
superlatives). To make these roles more generic, we only add `-91` as a token so the model can learn to generate them.
This format is the same as for senses (which end in e.g. `-01` for sense 01). So technically, a sense `-91`
can be ambiguous with a special frame that ends in `-91`. We will assume that `-91` always indicates a special frame's 
role and never a sense because a sense 91 is probably quite rare.

It can also occur as part of, e.g., a phone number. Unfortunately we have no solution for that ambiguity so when
tokenizing there might be inconsistencies in how spaces are handled after the token as `-91` will now always have
a space after it but no space before it.


## Quantity

AMR offers a wide range of [quantity specifiers](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#quantities).
These specifiers always end with `-quantity`, so to make this more generic, we only add `-quantity` as a token.

# Other entities

Some [other entities](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#other-entities-dates-times-percentages-phone-email-urls)
are also allowed, such as `date-entity` or `phone-number-entity`. To make this more generic, we only add `-entity`.

# Special cases

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

# Sentences

AMR allows multi-sentences constructions (which can also be phrases). Sentences are denoted with `:sntX`. We add
up to `:snt9`. Because it occurs often, we also add the special token `multi-sentence`.

# Senses

AMR tracks different senses of words with OntoNotes specifiers, e.g. `charge-05`. We use `:senseX` instead.

Note the difference with other numbers arguments that we add. Here, the sense ID is the same as the sense ID in the 
corpus, meaning that it counts `00`, `01`, `02` etc, and not `0` `1` `2`. Therefore, we also add `:sense1` in addition
`:sense01` so that the model can also generated `:sense12` automatically, based on `:sense1` + `2`.

# Reference tokens and their IDs

In AMR, we often refer back to other concepts by means of variables. We us `:refX` as variables and add up to 9 
(e.g. `:ref9`). Higher numbers can be generated dynamically.

# Branch boundaries

Because we are predicting a linearized version of AMR, we need a way to encapsulate branches, i.e., to indicate when
a branch starts and stops. In AMR, these branches are typically relations. We therefore add `:startrel` and `:endrel`
to indicate the start and end of a relation branch.

# Subsets

[Subsets](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#subsets) are covered with `:subset`, which we add.
Due to the generic token `-of` that is in the vocabulary, `:subset-of` can automatically be generated.

# Tree start/end

We will assume that the start and end of the prediction indicate the start and end of the tree. So no special tokens
are needed there. However, MBART relies on special language tokens to learn language-specific phenomena (and even
translate). Therefore, we add the special token `amr_XX` which we will use as a special token.