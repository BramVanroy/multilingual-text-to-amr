# multilingual-text-to-amr
 
An adaptation of MBART to parse text into AMR for multiple languages.

## AMR guidelines

https://github.com/amrisi/amr-guidelines/blob/master/amr.md

## Language modeling head

Unlike Spring, we use a learned LM head. Spring uses the shared weight of the embeddings to predict the tokens.

## LICENSE

Part of this code (especially the graph processing) was adapted from [SPRING](https://github.com/SapienzaNLP/spring).
Their cc-by-nc-sa-4.0 license is included in the third_party directory.

Our own contributions are distributed under a GPLv3 [license](LICENSE).
