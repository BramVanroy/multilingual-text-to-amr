# TODO AMR
  
- Test tokenization
    - tokenize text of AMR <> detokenize -> make sure they are the same
    - linearize, tokenize <> detokenize, delinearize -> assert they are the same


Back to the Drawing Board AGAIN
- Only add prefixes (:ARG, :snt, :ref, :op) and not the number roles
- Add :startrel, :endrel, :startlit, :endlit
- Add ":rel-of" for -of in relations
- Add "other roles" (compare https://github.com/SapienzaNLP/spring/blob/main/data/vocab/additions.txt#L179 and https://github.com/BramVanroy/multilingual-text-to-amr/blob/main/src/mbart_amr/data/tokens.py#L19)
- in postprocessing, just glue things together that should not have spaces between them, both :ARG + number, of tokens
  - for roles + num: glue together
  - for other tokens: if one of them starts/ends with "-", glue together; if not, wrap in LIT