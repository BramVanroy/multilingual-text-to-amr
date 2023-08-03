import importlib
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, AutoConfig


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "the directory containing the adapters"})
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
        model = arch_cls.from_pretrained(script_args.model_name_or_path, device_map="cpu")

        # Make sure that the model can generate, if that is required
        if not script_args.no_require_generate and not model.can_generate():
            print(f"Model arch {arch} does not implement generate(), which we require. Skipping...")
            continue

        # is_main_process=False -> do not save the config
        # The config should already exist. Overwriting it, may change some directory values in it
        model.save_pretrained(script_args.output_name, safe_serialization=True, is_main_process=False)
        saved = True
        break

    if not saved:
        print("Did not find any architecture that was compatible with .generate() and therefore could not correctly"
              " save the model. Is something wrong with your config?")
    else:
        print(f"Safetensors saved to {script_args.output_name}!")
