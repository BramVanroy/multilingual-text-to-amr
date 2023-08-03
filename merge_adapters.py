"""Modified from https://github.com/lvwerra/trl/blob/main/examples/stack_llama/scripts/merge_peft_adapter.py"""
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, HfArgumentParser, AutoConfig


@dataclass
class ScriptArguments:
    adapter_model_name: str = field(metadata={"help": "the directory containing the adapters"})
    base_model_name: str = field(metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "where to save the output"})
    no_require_generate: bool = field(default=False,
                                      metadata={"help": "whether to NOT require that the model implements"
                                                        " .generate(). This is useful if you are trying to convert a"
                                                        " model that is not a generative one (e.g. for classification)"})


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.output_name is None:
        script_args.output_name = script_args.model_name_or_path

    # Get the config
    config = AutoConfig.from_pretrained(script_args.model_name_or_path)

    saved = False
    # Get related architectures for this model's config
    for arch in config.architectures:
        # Dynamically load the instantiation class and init model
        arch_cls = getattr(importlib.import_module("transformers"), arch)
        model = arch_cls.from_pretrained(script_args.model_name_or_path, device_map="cpu", trust_remote_code=True)
        # Make sure that the model can generate, if that is required
        if not script_args.no_require_generate and not model.can_generate():
            print(f"Model arch {arch} does not implement generate(), which we require. Skipping...")
            continue

        peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
        tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name, trust_remote_code=True)

        # Load the LoRA model
        model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
        model.eval()

        model = model.merge_and_unload()

        model.save_pretrained(f"{script_args.output_name}", safe_serialization=True)
        tokenizer.save_pretrained(f"{script_args.output_name}")
        saved = True
        break

    if not saved:
        print("Did not find any architecture that was compatible with .generate() and therefore could not correctly"
               " save the model. Is something wrong with your config?")
    else:
        print(f"Merged and saved to {str(Path(script_args.output_name).resolve())}!")
