from mbart_amr.data.tokenization import AMRMBartTokenizer

from transformers import LogitsProcessor


class AMRLogitsProcessorBase(LogitsProcessor):
    def __init__(self, tokenizer: AMRMBartTokenizer, max_length: int, debug: bool = False):
        self.tokenizer = tokenizer
        self.debug = debug

        # We need -1 because there is another logitprocessor (? somewhere)
        # that ensures that the last token is EOS, so we account for that
        self.max_length = max_length - 1

        super().__init__()

    def _debug_decode(self, input_ids, skip_special_tokens=False):
        return self.tokenizer.decode_and_fix(input_ids, skip_special_tokens=skip_special_tokens)
