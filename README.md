# multilingual-text-to-amr
 
An adaptation of MBART to parse text into AMR for multiple languages.

## AMR guidelines

https://github.com/amrisi/amr-guidelines/blob/master/amr.md

## TODO

- postprocessing for invalid trees that the model may produce;
- add smatch as metric;
- add script to only run prediction on an already trained model so that we can go from text -> AMR. Must include postprocessing!
- smart initialization of our special tokens by copying weights from the existing embedding to the new tokens;
- add conditional decoding in beam search (if time)
- add hyperparameter search (maybe)

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
