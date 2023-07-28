import torch
from multi_amr.constraints.base import AMRLogitsProcessorBase
from multi_amr.data.tokenization import AMRMBartTokenizer


class FirstTokenProcessor(AMRLogitsProcessorBase):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int, debug: bool = False):
        super().__init__(tokenizer, max_length, debug)
        # Allow :ref1, multi-sentence, amr-unknown, amr-choice
        self.allowed_tokens_in_first_position = torch.LongTensor(
            [
                self.tokenizer.multisent_idx,
                self.tokenizer.ref_idxs[0].item(),
                self.tokenizer.unknown_idx,
                self.tokenizer.choice_idx,
            ]
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # The shape is also influenced by the num_beams, which is the first dimension. E.g., [5, 7] for 5 beams
        for beam_idx in range(input_ids.size(0)):
            inputs = input_ids[beam_idx]
            logits = scores[beam_idx]
            num_inputs = inputs.size(0)

            """The tree cannot start with any special added tokens nor any special tokens (like <s>, which is already
            the first starting token).
            The first token CAN be a :ref, though!
            It can start with a special frame like "have-condition-91" but that is generated  generically with "have" 
            and does not depend on a special first token so that's fine.
            """
            if num_inputs == 1:
                mask = torch.cat((self.tokenizer.added_tokens_idxs, self.tokenizer.special_tokens_idxs))
                mask = mask[~torch.isin(mask, self.allowed_tokens_in_first_position)]
                logits[mask] = float("-inf")
                if self.debug:
                    print(f"num_inputs == 1\n{self._debug_decode(inputs)}\n")
                continue

        return scores
