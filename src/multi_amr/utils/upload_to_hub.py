"""Upload a given model to the HF hub."""
from dataclasses import dataclass, field

from huggingface_hub import HfApi
from transformers import HfArgumentParser


def upload(model_name: str, model_dir: str, username: str = "BramVanroy", exist_ok: bool = False):
    api = HfApi()
    api.create_repo(f"{username}/{model_name}", exist_ok=exist_ok)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=f"{username}/{model_name}",
        ignore_patterns=["**/*", "trainer_state.json", "training_args.bin"],
        repo_type="model",
    )


@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "model name to use on the hub"})
    model_dir: str = field(metadata={"help": "model directory where to upload from"})
    username: str = field(default="BramVanroy", metadata={"help": "model directory where to upload from"})
    exist_ok: str = field(
        default=False,
        metadata={"help": "if this flag is set, no error will be thrown when the remote repo already exists"},
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    upload(
        model_name=script_args.model_name,
        model_dir=script_args.model_dir,
        username=script_args.username,
        exist_ok=script_args.exist_ok,
    )


if __name__ == "__main__":
    main()
