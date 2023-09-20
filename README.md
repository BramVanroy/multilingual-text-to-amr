# Multilingual text to AMR parsing

**THIS REPOSITORY AND README IS STILL UNDER DEVELOPMENT. NOT READY FOR PRODUCTION**
 
An adaptation of MBART to parse text into AMR for multiple languages.

## Usage

First install the package by running the following in the root directory where `setup.py` is present.

```shell
pip install -e .
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
