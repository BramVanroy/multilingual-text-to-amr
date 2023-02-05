"""
Also read AMR guidelines
Parse corpus as linearized and check what's possible and what's not
---

2. Sense-ids only possible after tokens that are not special :ARG tokens and not double quotes
3. Opened `"` quotes need to be closed before a relation is started + open/close should match
4. REF can only follow after a role or after a startrel
5. Something about :wiki? (always followed by `"` or "-")
6. Sometimes :ARG can be followed by "-"
7. Check whether/which relations are always followed by a non-special token


"""
from collections import defaultdict

import torch

from mbart_amr.data.linearization import linearized2penmanstr
from mbart_amr.data.tokenization import AMRMBartTokenizer
from transformers import MBartForConditionalGeneration, LogitsProcessor, LogitsProcessorList

from mbart_amr.data.tokens import STARTREL, ENDREL, LANG_CODE, SENSES, ROLES
from mbart_amr.utils import ends_in_valid_role


class AMRLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int):
        self.start_idx = tokenizer.convert_tokens_to_ids(STARTREL)
        self.end_idx = tokenizer.convert_tokens_to_ids(ENDREL)
        self.lang_idx = tokenizer.convert_tokens_to_ids(LANG_CODE)
        self.tokenizer = tokenizer
        self.sense_idxs = tokenizer.convert_tokens_to_ids(SENSES)
        self.role_idxs = tokenizer.convert_tokens_to_ids(ROLES)
        self.dbl_quote_prespace = tokenizer.convert_tokens_to_ids('▁"')
        self.dbl_quote = tokenizer.convert_tokens_to_ids('"')

        # We need -1 because there is another logitprocessor (? somewhere)
        # that ensures that the last token is EOS, so we account for that
        self.max_length = max_length-1
        self.voc_size = len(tokenizer)
        self.voc_mask = torch.arange(self.voc_size)

        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # The shape is also influenced by the num_beams, which is the first dimension. E.g., [5, 7] for 5 beams
        for beam_idx in range(input_ids.size(0)):
            inputs = input_ids[beam_idx]
            seq_length = inputs.size(0)
            logits = scores[beam_idx]

            prev_valid_role = ends_in_valid_role(self.tokenizer, inputs)

            uniq_counts = defaultdict(int)  # Counter that will default to 0 for unknown keys
            # torch.unique returns a tuple, the sorted inp tensor, and a tensor with the frequencies
            # then, map these tensors to a list and zip them into a dict to get {input_id: frequency}
            # By updating the `counter`, we have a frequency dictionary and default values of 0
            uniq_counts.update(dict(zip(*map(torch.Tensor.tolist, inputs.unique(return_counts=True)))))

            diff_start_end = uniq_counts[self.start_idx] - uniq_counts[self.end_idx]

            # Can't generate a closing tag if we have no "open" tag
            if uniq_counts[self.start_idx] == uniq_counts[self.end_idx]:
                logits[self.end_idx] = float("-inf")

            # Can't generate a close tag directly after an open tag
            # UNLESS this is the last token to predict, so we just try to catch
            # the exceptional case that the linearization ends in a meaningless [:startrel :endrel] combo
            if inputs[-1] == self.start_idx and not seq_length == self.max_length-1:
                logits[self.end_idx] = float("-inf")

            # Can only generate start_idx, if the previous token was a valid role
            if not prev_valid_role:
                logits[self.start_idx] = float("-inf")

            # If the current seq length + the mismatch between start/end equals the
            # max sequence length, then the following token(s) should all be ENDs
            if seq_length + diff_start_end >= self.max_length:
                mask = self.voc_mask[self.voc_mask != self.end_idx]
                logits[mask] = float("-inf")

                # TODO: probably want to "ramp" this up
                # e.g., slowly give more weight to END tags by multiplying logit of END with
                # a constant. The question is, how do we define the value for that constant

            # Can't predict sense IDs if the previous token was a "special" kind of token
            # TODO: they can also not occur after double quotes
            if inputs[-1] in self.tokenizer.added_tokens_encoder.values() or prev_valid_role:
                logits[self.sense_idxs] = float("-inf")

        return scores


def main():
    tokenizer = AMRMBartTokenizer.from_pretrained("BramVanroy/mbart-en-to-amr", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained("BramVanroy/mbart-en-to-amr")

    model.resize_token_embeddings(len(tokenizer))

    gen_kwargs = {"max_length": 36, "num_beams": 1}

    if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
        gen_kwargs["max_length"] = model.config.max_length
    gen_kwargs["num_beams"] = (
        gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else model.config.num_beams
    )
    gen_kwargs["logits_processor"] = LogitsProcessorList([AMRLogitsProcessor(tokenizer, gen_kwargs["max_length"])])

    encoded = tokenizer("Should we go home now ?", return_tensors="pt")
    generated = model.generate(
            **encoded,
            **gen_kwargs,
        )

    decoded = tokenizer.decode_and_fix(generated[0])[0]
    print(generated)
    print(decoded)
    print(linearized2penmanstr(decoded))


if __name__ == '__main__':
    main()