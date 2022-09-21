"""Pretraining models for denoising multilingual language modeling on a text file or a dataset.
"""
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import transformers
from amr_bart.amr_bart.trainer_amr_bart import (AMRTrainer, compute_metrics,
                                                preprocess_logits_for_metrics)
from amr_bart.data.dataset_amr_bart import AMRDataset, collate_amr
from amr_bart.utils.utils import instantiate_model_and_tokenizer
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          is_torch_tpu_available, set_seed)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/bart-large",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    from_pretrained: bool = field(
        default=True,
        metadata={"help": "Whether to finetune a pretrained model or train from scratch."},
    )
    additional_tokens_smart_init: bool = field(
        default=True,
        metadata={"help": "Whether to automatically initialize new, special AMR tokens."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_directory: Optional[str] = field(
        default=None,
        metadata={"help": "Directory that contains training data. Will recursively" " be traversed for *.txt files"},
    )
    validation_directory: Optional[str] = field(
        default=None,
        metadata={"help": "Directory that contains validation data. Will" " recursively be traversed for *.txt files"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    mask_ratio: float = field(
        default=0.3, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    random_ratio: float = field(
        default=0.1, metadata={"help": "The probability with which to (randomly) replace a token by a random token"}
    )
    insert_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "The probability with which to (randomly) insert noise. Will add `insert_ratio * input.numel()` noise"
            )
        },
    )
    rotate_ratio: float = field(
        default=0.0,
        metadata={
            "help": "The probability with which to (randomly) add rolling noise (i.e., shifted tokens in order)"
        },
    )
    permute_sentence_ratio: float = field(
        default=1.0, metadata={"help": "Ratio of sentences to be permuted in each document"}
    )
    poisson_lambda: float = field(
        default=3.5, metadata={"help": "Mean of Poisson distribution used to generate span-lengths to be masked"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    ############################
    # Load tokenizer and model #
    ############################
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.tokenizer_name and model_args.model_name_or_path:
        model_args.tokenizer_name = model_args.model_name_or_path

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.config_name and model_args.model_name_or_path:
        model_args.config_name = model_args.model_name_or_path

    model, tokenizer = instantiate_model_and_tokenizer(
        model_args.model_name_or_path,
        model_args.config_name,
        model_args.tokenizer_name,
        additional_tokens_smart_init=model_args.additional_tokens_smart_init,
        from_pretrained=model_args.additional_tokens_smart_init,
        tokenizer_kwargs=tokenizer_kwargs,
        config_kwargs=config_kwargs,
    )

    #######################
    # Load datasets #
    #######################
    train_files = list(Path(data_args.train_directory).rglob("*.txt"))
    validation_files = list(Path(data_args.validation_directory).rglob("*.txt"))

    train_dataset = AMRDataset(train_files, tokenizer)
    validation_dataset = AMRDataset(validation_files, tokenizer)

    # Initialize our Trainer
    training_args.remove_unused_columns = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=partial(collate_amr, tokenizer),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(validation_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(validation_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text2text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
