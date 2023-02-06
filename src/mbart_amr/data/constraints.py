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

from mbart_amr.data.tokens import STARTREL, ENDREL, LANG_CODE, SENSES, ROLES, STARTLIT, ENDLIT, REFS
from mbart_amr.utils import ends_in_valid_role, ends_in_valid_ref


class AMRLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int):
        self.tokenizer = tokenizer

        self.start_rel_idx = tokenizer.convert_tokens_to_ids(STARTREL)
        self.end_rel_idx = tokenizer.convert_tokens_to_ids(ENDREL)
        self.start_lit_idx = tokenizer.convert_tokens_to_ids(STARTLIT)
        self.end_lit_idx = tokenizer.convert_tokens_to_ids(ENDLIT)
        self.end_lit_idx = tokenizer.convert_tokens_to_ids(ENDLIT)
        self.lang_idx = tokenizer.convert_tokens_to_ids(LANG_CODE)
        self.sense_idxs = torch.LongTensor(tokenizer.convert_tokens_to_ids(SENSES))
        self.role_idxs = torch.LongTensor(tokenizer.convert_tokens_to_ids(ROLES))
        self.ref_idxs = torch.LongTensor(tokenizer.convert_tokens_to_ids(REFS))

        # We need -1 because there is another logitprocessor (? somewhere)
        # that ensures that the last token is EOS, so we account for that
        self.max_length = max_length - 1
        self.voc_size = len(tokenizer)
        self.special_tokens_idxs = torch.LongTensor(self.tokenizer.all_special_ids)
        self.added_tokens_idxs = torch.LongTensor(list(self.tokenizer.added_tokens_encoder.values()))
        self.voc_idxs_for_mask = torch.arange(self.voc_size)

        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # The shape is also influenced by the num_beams, which is the first dimension. E.g., [5, 7] for 5 beams
        for beam_idx in range(input_ids.size(0)):
            inputs = input_ids[beam_idx]
            logits = scores[beam_idx]
            num_inputs = inputs.size(0)

            # 1. STARTING TOKEN
            """The tree cannot start with any special added tokens nor any special tokens (like <s>)
            The first token is already a <s> token.
            It can start with a special frame like "have-condition-91" but that is generated
            generically with "have" and does not depend on a special first token so that's fine
            """
            if num_inputs == 1:
                mask = torch.cat((self.added_tokens_idxs, self.special_tokens_idxs))
                logits[mask] = float("-inf")
                continue

            seq_length = inputs.size(0)
            last_item = inputs[-1].item()

            # -- collect unique counts in the current inputs for each token ID
            uniq_counts = defaultdict(int)  # Counter that will default to 0 for unknown keys
            # torch.unique returns a tuple, the sorted inp tensor, and a tensor with the frequencies
            # then, map these tensors to a list and zip them into a dict to get {input_id: frequency}
            # By updating the `counter`, we have a frequency dictionary and default values of 0
            uniq_counts.update(dict(zip(*map(torch.Tensor.tolist, inputs.unique(return_counts=True)))))

            # 2. LENGTH-RELATED
            """Because of the restrictions that we have due to a max_length
            we may need to force close some structures to make sure the output is valid"""
            diff_start_end_rel = uniq_counts[self.start_rel_idx] - uniq_counts[self.end_rel_idx]
            # TODO

            # 3. OPEN/CLOSING tags
            """REL and LIT have some specific restrictions to make sure that these structural elements
            are opened and closed consistently."""
            # 3.1. RELs
            # Can't generate a closing rel tag if we have no "open" tag
            if uniq_counts[self.start_rel_idx] == uniq_counts[self.end_rel_idx]:
                logits[self.end_rel_idx] = float("-inf")

            # Can't generate a close rel tag directly after an open tag
            # UNLESS this is the last token to predict, so we just try to catch
            # the exceptional case that the linearization ends in a meaningless [:startrel :endrel] combo
            if last_item == self.start_rel_idx and not seq_length == self.max_length - 1:
                logits[self.end_rel_idx] = float("-inf")

            # 3.1. LITs
            # Can't generate a closing lit tag if we have no "open" tag
            if uniq_counts[self.start_lit_idx] == uniq_counts[self.end_lit_idx]:
                logits[self.end_lit_idx] = float("-inf")
            # Unlike in REL, we cannot open multiple embedded LITs, so open lit not possible if another is open
            if uniq_counts[self.start_lit_idx] > uniq_counts[self.end_lit_idx]:
                logits[self.start_lit_idx] = float("-inf")

            # 4. ALLOWED TOKEN ORDERS
            """Many tokens can only occur after specific tokens"""
            prev_ends_in_valid_role = ends_in_valid_role(self.tokenizer, inputs)
            prev_ends_in_valid_ref = ends_in_valid_ref(self.tokenizer, inputs)

            # Can only generate start_rel_idx, if the previous token was a valid role
            if not prev_ends_in_valid_role:
                logits[self.start_rel_idx] = float("-inf")

            # Can only generate a ref idx, if the previous token was a valid role or :startrel
            if not (prev_ends_in_valid_role and last_item != self.start_rel_idx):
                logits[self.ref_idxs] = float("-inf")

            # Can't predict sense IDs if the previous token was a "special" kind of token, a role, or a ref
            if last_item in self.added_tokens_idxs or prev_ends_in_valid_role or prev_ends_in_valid_ref:
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
