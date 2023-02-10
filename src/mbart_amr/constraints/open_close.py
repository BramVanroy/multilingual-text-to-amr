import torch
from mbart_amr.constraints.base import AMRLogitsProcessorBase, input_ids_counts
from mbart_amr.data.tokenization import AMRMBartTokenizer


class OpenCloseTokenProcessor(AMRLogitsProcessorBase):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int, debug: bool = False):
        super().__init__(tokenizer, max_length, debug)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for beam_idx in range(input_ids.size(0)):
            inputs = input_ids[beam_idx]
            logits = scores[beam_idx]

            last_item = inputs[-1].item()
            uniq_counts = input_ids_counts(inputs)

            """REL and LIT have some specific restrictions to make sure that these structural elements
            are opened and closed consistently."""
            # RELs: can't generate a closing rel tag if we have no "open" tag
            if uniq_counts[self.start_rel_idx] == uniq_counts[self.end_rel_idx]:
                logits[self.end_rel_idx] = float("-inf")
                if self.debug:
                    print(
                        f"start_rel_idx == end_rel_idx\tDISABLE: {self._debug_decode([self.end_rel_idx])}\n"
                        f"{self._debug_decode(inputs)}"
                    )

            # Can't generate a close rel tag directly after an open tag
            if last_item == self.start_rel_idx:
                logits[self.end_rel_idx] = float("-inf")
                if self.debug:
                    print(
                        f"last_item == start_rel_idx\tDISABLE: {self._debug_decode([self.end_rel_idx])}\n"
                        f"{self._debug_decode(inputs)}"
                    )

            # LITs: Can't generate a closing lit tag if we have no "open" tag
            if uniq_counts[self.start_lit_idx] == uniq_counts[self.end_lit_idx]:
                logits[self.end_lit_idx] = float("-inf")
                if self.debug:
                    print(
                        f"start_lit_idx == end_lit_idx\tDISABLE: {self._debug_decode([self.end_lit_idx])}\n"
                        f"{self._debug_decode(inputs)}"
                    )

            # Unlike in REL, we cannot open multiple embedded LITs, so open lit not possible if another is open
            if uniq_counts[self.start_lit_idx] > uniq_counts[self.end_lit_idx]:
                logits[self.start_lit_idx] = float("-inf")
                if self.debug:
                    print(
                        f"start_lit_idx > end_lit_idx\tDISABLE: {self._debug_decode([self.start_lit_idx])}\n"
                        f"{self._debug_decode(inputs)}"
                    )

            # Cannot generate any special tokens as long as LIT is open (except for closing LIT)
            if uniq_counts[self.start_lit_idx] > uniq_counts[self.end_lit_idx]:
                mask = torch.cat((self.added_tokens_idxs, self.special_tokens_idxs))
                mask = mask[~torch.isin(mask, self.end_lit_idx)]
                logits[mask] = float("-inf")
                if self.debug:
                    print(
                        f"start_lit_idx > end_lit_idx\tDISABLE: {self._debug_decode([self.start_lit_idx])}\n"
                        f"{self._debug_decode(inputs)}"
                    )

        return scores
