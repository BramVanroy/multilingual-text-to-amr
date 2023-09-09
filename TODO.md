# Spring notes
- Batch size is given in tokens (500). This comes down to an Average batch size of 3.33 sequences BUT they also accumulate for 10 steps (so total batch size of around 32)
- They use dropout=0.25 and attention_dropout=0.0 (in BART)
- Looking at the code, it seems that they train for 30 epoch (not 250k steps). Steps are only used for the lr scheduler

# TODO
- 
- check todos in code
- probably use dataset instead of preprocess? (but caching...)
- OPTIONALLY make it possible to choose whether to include all pointers, or only when a pointer is referred to. That would greatly reduce the sequence length
- 

# Difficulties/changes
- Tokenizers not correctly tokenizing a token: not adding a space to a token that is after a special token. So "<pointer:1> hello" will still be tokenized without prefix space --> THIS MAKES M\OUR APPROACH IMPOSSIBLE...
- "–" != "-" (en-dash): some tokenizers do not support it, but if you swap it out on the prediction side (and not on the source), then this will impact score (like 96% match instead of 100)

# Results
## SPRING results
- Original (reproduced) 0.830 with regular smatch (same as in paper, table 5)
- With smatchpp: {'F1': {'result': 82.86}, 'Precision': {'result': 82.03}, 'Recall': {'result': 83.7}}}
