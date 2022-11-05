# Added tokens

Sources:

- [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md)
- [AMR dictionary](https://amr.isi.edu/doc/amr-dict.html)

## Relations and roles

The list of added tokens is mostly inspired by the [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#part-ii--concepts-and-relations).

For `:ARGX` we add up to `:ARG10`. For `:opX` up to `:op10`. This does not mean that higher ranks, such as `:op25`, 
are not possible. But it does mean that they are not based on a single token but need to be generated. Therefore,
we also add `:op` so that the model can still generate outliers like `:op25` with multiple subtokens, i.e. `:op`
and `25`. The same is true for `:ARG`, which we add.

## Entity types (not added)

AMR describes specific entity types in [the guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#named-entities).
However, because these seem semantically already related to "real" words, we do not explicitly add these as whole
tokens in hopes that the language model already has a good "understanding" of these tokens.

## Special frames for roles

Some special frames are possible in AMR. These always end with `-91`, e.g. `have-degree-91` (for comparatives and
superlatives). To make these roles more generic, we only add `-91` as a token so the model can learn to generate them.

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
- All relations have an inverse relation that is created with the `-of` suffix, e.g., `:ARG0-of`. We therefore add `-of` as a token.
- Questions in AMR are denoted with the special relation `amr-unknown`, which we add. (It is therefore similar to `?`)
- Choice questions are a bit different, they are marked with `amr-choice`, which we add. (Similar to `or`)
- Negative polarity `:polarity -` is a way of negating what is being said. This is so frequent that we add a special
`:negation` token. (Similar to `not`)

# Sentences

AMR allows multi-sentences constructions (which can also be phrases). Sentences are denoted with `:sntX`. We add
up to `:snt10` and also add a generic `:snt`.

# Senses

AMR tracks different senses of words with OntoNotes specifiers, e.g. `charge-05`. We add 100 special `:senseX` tokens,
and add `:sense` for generalizability. That means the model can still generate `:sense200` with multiple tokens.
`:senseNO` indicates that no sense was indicated for a given word. Note that counting is with a prepended zero for
`< 10`. E.g. `:sense07`.

# Reference tokens and their IDs

In AMR, we often refer back to other concepts by means of variables. That means that we need a way to keep track of
the referent and referees. To distingusih them, we use `:termX` for antecedents, and `:refX` for items that refer to 
term `X`. E.g., `:term5` is the antecedent that `:ref5` refers to. Again, we include up to `:term100` and `:ref100`
but also include a generic `:term` and `:ref`.

# Branch boundaries

Because we are predicting a linearized version of AMR, we need a way to encapsulate branches, i.e., to indicate when
a branch starts and stops. In AMR, these branches are typically relations. We therefore add `:startrel` and `:endrel`
to indicate the start and end of a relation branch.

# Tree start/end

We will assume that the start and end of the prediction indicate the start and end of the tree. So no special tokens
are needed there. However, MBART relies on special language tokens to learn language-specific phenomena (and even
translate). Therefore, we add the special token `amr_XX` which we will use as a special token.