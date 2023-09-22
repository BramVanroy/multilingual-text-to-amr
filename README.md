# Multilingual text to AMR parsing

- MBART finetuned [models](https://huggingface.co/collections/BramVanroy/multilingual-text-to-amr-650b0fd576856b9acb257535) for English, Spanish and Dutch
- [Demo](https://huggingface.co/spaces/BramVanroy/text-to-amr) illustrating the text-to-AMR capabilities

## Install

First install the package by cloning and installing. For now, checkout the alpha release that should be compatible with commands
presented here.

```shell
git clone https://github.com/BramVanroy/multilingual-text-to-amr.git
cd multilingual-text-to-amr
git checkout tags/v1.0.0-alpha
pip install .
```

## Usage

### CLIN models

To reproduce the models that are presented at CLIN, you can find the related configuration files under `configs/`. You can then train the model
accordingly by using the config that you want, optionally overwriting some of the CLI arguments:

```python
python src/multi_amr/run_amr_generation.py <configs/config.json> <overriding_args>
```

## Notes

# Spring notes

- Batch size is given in tokens (500). This comes down to an Average batch size of 3.33 sequences BUT they also accumulate for 10 steps (so total batch size of around 32)
- They use dropout=0.25 and attention_dropout=0.0 (in BART -- not MBART)
- Looking at the code, it seems that they train for 30 epoch (not 250k steps). Steps are only used for the lr scheduler
- Original (reproduced) 0.830 with regular smatch (same as in paper, table 5)
- With smatchpp: {'F1': {'result': 82.86}, 'Precision': {'result': 82.03}, 'Recall': {'result': 83.7}}}

# Difficulties/changes
- Tokenizers of different models behave differently, especially considering space characters
- "–" != "-" (en-dash): some tokenizers do not support it, but if you swap it out on the prediction side (and not on the source), then this will impact score (like 96% match instead of 100)
- Poorly encoded data " You re not sittin  there in a back alley and sayin  hey what do you say, five bucks?"
