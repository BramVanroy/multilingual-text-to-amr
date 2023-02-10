import torch
from mbart_amr.constraints.base import AMRLogitsProcessorBase
from mbart_amr.constraints.functions import ends_in_prep, ends_in_role
from mbart_amr.data.tokenization import AMRMBartTokenizer


class AllowedTokensProcessor(AMRLogitsProcessorBase):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int, debug: bool = False):
        super().__init__(tokenizer, max_length, debug)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # The shape is also influenced by the num_beams, which is the first dimension. E.g., [5, 7] for 5 beams
        for beam_idx in range(input_ids.size(0)):
            inputs = input_ids[beam_idx]
            logits = scores[beam_idx]
            num_inputs = inputs.size(0)
            last_item = inputs[-1].item()

            # :start_rel can only follow valid role
            if not ends_in_role(inputs, self.tokenizer):
                logits[self.tokenizer.start_rel_idx] = float("-inf")
                if self.debug:
                    print(f"not ends_in_role\n{self._debug_decode(inputs)}\n")

            # :refXX can only follow :start_rel or a role...
            if not (last_item == self.tokenizer.start_rel_idx or ends_in_role(inputs, self.tokenizer)):
                mask = self.tokenizer.ref_idxs
                # ... except for :ref1 which can occur at the start of the sequence
                if num_inputs == 1:
                    mask = mask[~torch.isin(mask, self.tokenizer.ref_idxs[0])]

                logits[mask] = float("-inf")
                if self.debug:
                    print(f"not (last_item == start_rel or ends_in_role)\n{self._debug_decode(inputs)}\n")

            # :senseXX can only follow non-added and non-special tokens
            if last_item in self.tokenizer.special_tokens_idxs or last_item in self.tokenizer.added_tokens_idxs:
                logits[self.tokenizer.sense_idxs] = float("-inf")
                if self.debug:
                    print(
                        f"last_item in special_tokens_idxs or last_item in added_tokens_idxs\n{self._debug_decode(inputs)}\n"
                    )

            # ~~of can only follow roles (but not sents) or preps
            if not (
                ends_in_role(inputs, self.tokenizer, exclude_categories=["sents"])
                or ends_in_prep(inputs, self.tokenizer)
            ):
                logits[self.tokenizer.of_idx] = float("-inf")
                if self.debug:
                    print(f"not (ends_in_role (except sents) or ends_in_prep)\n{self._debug_decode(inputs)}\n")

            # ~~of cannot follow ~~of
            if last_item == self.tokenizer.of_idx:
                logits[self.tokenizer.of_idx] = float("-inf")
                if self.debug:
                    print(f"last_item == of_idx\n{self._debug_decode(inputs)}\n")

        return scores
