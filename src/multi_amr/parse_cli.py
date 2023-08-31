import dataclasses
import json
import sys
from pathlib import Path

from transformers import HfArgumentParser


def parse_cli(*arg_classes):
    """Parse command-line arguments. Can be read from a given JSON file, which in turn can be overridden
    by CLI options.
    """
    parser = HfArgumentParser(arg_classes)

    try:
        # Assumes that the first .json file is the config file (if any)
        config_file = next(iter(arg for arg in sys.argv if arg.endswith(".json")))
    except StopIteration:
        config_file = None

    run_name_specified = False
    if config_file:
        config_args = parser.parse_json_file(json_file=str(Path(config_file).resolve()))
        raw_config_json = json.loads(Path(config_file).read_text(encoding="utf-8"))

        config_arg_idx = sys.argv.index(config_file)
        other_args = sys.argv[config_arg_idx + 1 :]
        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}

        if "run_name" in arg_names or "run_name" in raw_config_json:
            run_name_specified = True

        required_args = [
            (act.option_strings[0], "dummy")
            for act in parser._actions
            if act.required and not any(act_s[2:] in arg_names for act_s in act.option_strings)
        ]
        required_args = [arg for req_dummy_args in required_args for arg in req_dummy_args]  # Flatten

        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(args=cli_args, look_for_args_file=False)

        all_args = []

        for cfg_dc, cli_dc in zip(config_args, cli_args):
            # Have to check explicitly for no_ for the automatically added negated boolean arguments
            # E.g. find_unused... vs no_find_unused...
            cli_d = {k: v for k, v in dataclasses.asdict(cli_dc).items() if k in arg_names or f"no_{k}" in arg_names}
            merged_args = dataclasses.replace(cfg_dc, **cli_d)

            # Normally, post_init of training_args sets run_name to output_dir
            # But if we overwrite output_dir with a CLI option, then we do not correctly update
            # run_name to the same value. Which in turn will lead to wandb to use the original "results/" as a run name
            # see: https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/integrations.py#L741-L742
            # Instead we explicitly have to set run_name to the output_dir again -- but of course only if the user
            # did not specifically specify run_name in the config or in the CLI
            if hasattr(merged_args, "output_dir") and hasattr(merged_args, "run_name") and not run_name_specified:
                merged_args = dataclasses.replace(merged_args, run_name=merged_args.output_dir)

            all_args.append(merged_args)
    else:
        all_args = parser.parse_args_into_dataclasses()

    return all_args
