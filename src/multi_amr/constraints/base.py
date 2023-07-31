from multi_amr.data.tokenization import AMRTokenizerWrapper
from transformers import LogitsProcessor

from multi_amr.utils import debug_decode


class AMRLogitsProcessorBase(LogitsProcessor):
    def __init__(self, tokenizer: AMRTokenizerWrapper, max_length: int, debug: bool = False):
        self.tokenizer = tokenizer
        self.debug = debug

        # We need -1 because there is another logitprocessor (? somewhere)
        # that ensures that the last token is EOS, so we account for that
        self.max_length = max_length - 1

        super().__init__()

    def _debug_decode(self, input_ids, skip_special_tokens=False):
        return debug_decode(input_ids, self.tokenizer, skip_special_tokens=skip_special_tokens)
