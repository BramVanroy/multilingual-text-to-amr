# Spring notes
- Batch size is given in tokens (500). This comes down to an Average batch size of 3.33 sequences BUT they also accumulate for 10 steps (so total batch size of around 32)
- They use dropout=0.25 and attention_dropout=0.0 (in BART)
- Looking at the code, it seems that they train for 30 epoch (not 250k steps). Steps are only used for the lr scheduler

# TODO to reset to Spring
- check whether input and labels passed to model are indeed the same (start/end token)
- patch default transformers BART to the spring variant and see if we get similar results? (not sure if worth it)
- implement wandb sweep for hyperparameter testing
- check if smatch calculation is the same

# NEXT
- looks like something might be going wrong in the delinearization or in 
