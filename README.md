# multilingual-text-to-amr
 
An adaptation of MBART to parse text into AMR for multiple languages.

## AMR guidelines

https://github.com/amrisi/amr-guidelines/blob/master/amr.md

## TODO

- postprocessing for invalid trees that the model may produce;
- smart initialization of our special tokens by copying weights from the existing embedding to the new tokens;
- dataset/pipeline improvements to allow multiple languages so that the model can learn to generate AMRs for different
languages
  - make sure that each batch contains data from the same language
  - make sure that a batch is randomly selected so that we train all languages in a mixed fashion. First a batch of
  language X than a batch of language Y, and so on

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
