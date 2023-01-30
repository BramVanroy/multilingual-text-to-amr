"""Finetuning MBart models on multilingual datasets for text-to-AMR"""
import dataclasses
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import evaluate as evaluate
import numpy as np
import smatch
import transformers
from amr import AMR
from mbart_amr.data.dataset import AMRDataset, collate_amr
from mbart_amr.data.linearization import linearized2penmanstr
from mbart_amr.data.tokenization import AMRMBartTokenizer
from mbart_amr.trainer import AMRTrainer, ExpandedSeq2SeqTrainingArguments
from mbart_amr.utils.smart_initialization import (freeze_encoder,
                                                  smart_initialization)
from transformers import (EarlyStoppingCallback, HfArgumentParser,
                          MBartForConditionalGeneration, set_seed)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/mbart-large-cc25",
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    src_langs: List[str] = field(
        metadata={
            "help": "A list of source languages that corresponds with the given order in"
            " `train|validation directories`. Make sure that the right language code is used with"
            " respect to the model that you are using."
        },
    )
    train_directories: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Directories that contains training data. Will recursively be traversed for *.txt files."
            " One directory per source language."
        },
    )
    validation_directories: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Directory that contains validation data. Will recursively be traversed for *.txt files."
            " One directory per source language."
        },
    )
    test_directories: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Directory that contains test data. Will recursively be traversed for *.txt files."
            " One directory per source language. Will only be used when using predict functionality."
        },
    )
    test_data_has_labels: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether or not the .txt files in the test_directories contain AMR data (so :sent sentences as well as"
                "AMR graphs) or not (plain text files). The predictions will be written to an output file"
                " test_predictions.txt. If AMR labels, the evaluation scores will also be printed."
            )
        },
    )
    input_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model. Batches will be padded up to max."
                " length in the batch."
            )
        },
    )
    output_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total output sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model. Batches will be padded up to max."
                " length in the batch."
            )
        },
    )
    max_train_samples_per_language: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set per given language."
            )
        },
    )
    max_eval_samples_per_language: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set per given language."
            )
        },
    )
    max_test_samples_per_language: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of test examples to this "
                "value if set per given language."
            )
        },
    )
    save_amrs: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to save (in)valid generated AMRs to a file 'invalid-amrs.txt' in the output directory."
                " During prediction ('--do_predict') predictions are written to 'generated_predictions_{lang}'"
                " regardless of this flag."
            )
        },
    )
    remove_wiki: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to remove the special 'wiki:' tags from the AMR.")},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExpandedSeq2SeqTrainingArguments))

    try:
        # Assumes that the first .json file is the config file (if any)
        config_file = next(iter(arg for arg in sys.argv if arg.endswith(".json")))
    except StopIteration:
        config_file = None

    if config_file:
        config_args = parser.parse_json_file(json_file=os.path.abspath(config_file))

        # Find all other CLI arguments, i.e. the ones that follow the config file
        config_arg_idx = sys.argv.index(config_file)
        other_args = sys.argv[config_arg_idx + 1 :]

        # Find the argument names on CLI, i.e., those starting with -- (e.g., `output_dir`, or `fp16`)
        # arg[2:] to remove "--"
        arg_names = {arg[2:] for arg in other_args if arg.startswith("--")}

        # Get the required arguments from the parser so that we can generate dummy values
        # This is needed for the next step, otherwise we get "xxx is a required argument" if we
        # do not specify the required argument on the CLI
        # We assume here that all "required" fields require one single argument, here just a dummy
        # Do NOT generate the dummy argument if it is already given as an arg in arg_names
        required_args = [
            (act.option_strings[0], "dummy")
            for act in parser._actions
            if act.required and not any(act_s[2:] in arg_names for act_s in act.option_strings)
        ]
        required_args = [arg for req_dummy_args in required_args for arg in req_dummy_args]  # Flatten

        # Parse the `cli_args` (actual CLI args + dummy required args) into dataclasses
        cli_args = other_args + required_args
        cli_args = parser.parse_args_into_dataclasses(args=cli_args, look_for_args_file=False)

        # Iterate over couples of dataclasses, where the first one is the initial one from config
        # and the second one is the one with CLI args + dummy required arguments
        # In this loop, we replace values in cfg_dc with the ones from the CLI
        # but only if they were _real_ CLI arguments (not the required dummies we generated)
        all_args = []
        for cfg_dc, cli_dc in zip(config_args, cli_args):
            # Filter the loaded `other_args` to only the arguments that were specified on the command-line (`arg_names`)
            cli_d = {k: v for k, v in dataclasses.asdict(cli_dc).items() if k in arg_names}
            # Replace the values in the config-loaded args with the ones loaded from cli
            all_args.append(dataclasses.replace(cfg_dc, **cli_d))

        model_args, data_args, training_args = all_args
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
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.tokenizer_name and model_args.model_name_or_path:
        model_args.tokenizer_name = model_args.model_name_or_path

    tokenizer = AMRMBartTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs, src_lang="en_XX")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.config_name and model_args.model_name_or_path:
        model_args.config_name = model_args.model_name_or_path

    model = MBartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    model.resize_token_embeddings(len(tokenizer))

    if training_args.smart_initialization:
        if Path(model_args.model_name_or_path).exists() or (training_args.do_eval and not training_args.do_train):
            logger.warning(
                "You have enabled smart initialization but you seem to be loading a model that you have"
                " already trained. This may lead to worse-than-expected performance because you will be"
                " effectively overwriting the token embeddings of the added tokens"
            )
        model = smart_initialization(model, tokenizer, noise_range=training_args.noise_range)

    if training_args.freeze_encoder:
        model = freeze_encoder(model)

    #######################
    # CUSTOM METRICS #
    #######################
    def calculate_smatch(references: List[str], predictions: List[str]):
        total_match_num = total_test_num = total_gold_num = 0
        n_invalid = 0
        pdout = Path(training_args.output_dir)
        with pdout.joinpath("invalid-amrs.txt").open("a", encoding="utf-8") as fh_invalid, pdout.joinpath(
            "valid-amrs.txt"
        ).open("a", encoding="utf-8") as fh_valid:
            for sentid, (ref, pred) in enumerate(zip(references, predictions), 1):
                try:
                    ref_penman = linearized2penmanstr(ref)
                    # smatch uses its own AMR parser behind-the-scenes instead of penman, so we use it
                    # here to check if it's a valid tree. If note, this will return None
                    if AMR.parse_AMR_line(ref_penman) is None:
                        raise Exception
                except Exception:
                    n_invalid += 1
                    if data_args.save_amrs:
                        fh_invalid.write(f"REF_ERROR\t{datetime.now().time()}\t{ref}\n")
                    continue

                try:
                    pred_penman = linearized2penmanstr(pred)
                    if AMR.parse_AMR_line(pred_penman) is None:
                        raise Exception
                except Exception:
                    n_invalid += 1
                    if data_args.save_amrs:
                        fh_invalid.write(f"PRED_ERROR\t{datetime.now().time()}\t{pred}\n")
                    continue

                try:
                    best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
                        ref_penman, pred_penman, sent_num=sentid
                    )
                except Exception:
                    n_invalid += 1
                    # At this point, any error is probably caused by the prediction
                    if data_args.save_amrs:
                        fh_invalid.write(f"SMATCH_ERROR\t{datetime.now().time()}\t{pred_penman}\n")
                    continue

                total_match_num += best_match_num
                total_test_num += test_triple_num
                total_gold_num += gold_triple_num
                # clear the matching triple dictionary for the next AMR pair
                smatch.match_triple_dict.clear()
                # First the prediction, then the reference AMR
                if data_args.save_amrs:
                    fh_valid.write(f"{pred_penman}\n{ref_penman}\n\n")

            if n_invalid > 0:
                logger.warning(
                    f"{n_invalid:,} ({n_invalid / len(predictions) * 100:.2f}%) prediction(s) were not valid AMR. "
                    f" Smatch  scores only reflect the performance on valid AMR structures! Invalid structures have"
                    f" been appended to invalid-amrs.txt in the output directory."
                )

        score = smatch.compute_f(total_match_num, total_test_num, total_gold_num)

        return {
            "smatch_precision": score[0],
            "smatch_recall": score[1],
            "smatch_fscore": score[2],
            "ratio_invalid_amrs": n_invalid / len(predictions) * 100,
        }

    sb_metric = evaluate.load("sacrebleu")
    acc_metric = evaluate.load("accuracy")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # BLEU
        labels_for_bleu = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds_for_bleu = np.where(preds != -100, preds, tokenizer.pad_token_id)
        ref_linearizations = tokenizer.decode_and_fix(labels_for_bleu)
        pred_linearizations = tokenizer.decode_and_fix(preds_for_bleu)
        sb = {"bleu": sb_metric.compute(predictions=pred_linearizations, references=ref_linearizations)["score"]}

        smatch_score = calculate_smatch(ref_linearizations, pred_linearizations)

        # We can only calculate accuracy when we have the same number of predicted tokens and reference tokens
        # which is the case when predict_with_generate is false
        if not training_args.predict_with_generate:
            # Accuracy: flatten and calculate accuracy on flattened arrays
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            acc = acc_metric.compute(predictions=preds, references=labels)
            return {**acc, **sb, **smatch_score}

        return {**sb, **smatch_score}

    #######################
    # Load datasets #
    #######################
    train_dataset = None
    validation_dataset = None
    if training_args.do_train:
        train_dataset = AMRDataset(
            data_args.train_directories,
            src_langs=data_args.src_langs,
            remove_wiki=data_args.remove_wiki,
            max_samples_per_language=data_args.max_train_samples_per_language,
        )

    # Always validate during training
    # So with --do_train we also use the validation set
    # but with --do_eval we get a final performance of the best model on the validation set
    if training_args.do_train or training_args.do_eval:
        validation_dataset = AMRDataset(
            data_args.validation_directories,
            src_langs=data_args.src_langs,
            remove_wiki=data_args.remove_wiki,
            max_samples_per_language=data_args.max_eval_samples_per_language,
        )

    training_args.remove_unused_columns = False
    callbacks = []
    # If you want to use early stopping, both arguments have to be specified. Throw error if just one is specified.
    if training_args.early_stopping_patience is not None and training_args.early_stopping_threshold is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
                early_stopping_threshold=training_args.early_stopping_threshold,
            )
        )
    elif (training_args.early_stopping_patience is None or training_args.early_stopping_threshold is None) and not (
        training_args.early_stopping_patience is None and training_args.early_stopping_threshold is None
    ):
        raise ValueError(
            "Both 'early_stopping_patience' and 'early_stopping_threshold' must be given, or none of them."
            " If none are given, early stopping will not be used."
        )

    # Initialize our Trainer
    trainer = AMRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=partial(
            collate_amr,
            tokenizer=tokenizer,
            input_max_seq_length=data_args.input_max_seq_length,
            output_max_seq_length=data_args.output_max_seq_length,
        ),
        compute_metrics=compute_metrics,
        # if we use `predict_with_generate`, the returned values are the generated tokens
        # if we do not use `predict_with_generate`, the returned values are logits, so we need to to argmax
        # oursleves via the 'preprocess_logits_for_metrics' function
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if not training_args.predict_with_generate
        else None,
        callbacks=callbacks,
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

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text2text-generation"}

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

    # Evaluation
    max_length = training_args.generation_max_length
    num_beams = training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length,
            num_beams=num_beams,
            penalty_alpha=training_args.penalty_alpha,
            top_k=training_args.top_k,
            do_sample=training_args.do_sample,
            top_p=training_args.top_p,
        )

        # Will take max_eval_samples_per_language into consideration
        # because we never add more samples to the dataset than needed
        metrics["eval_samples"] = len(validation_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Loop over every language separately to avoid missing batches from the dataloader
        # and to keep track of the performance of each language separately
        for lang, directory in zip(data_args.src_langs, data_args.test_directories):
            test_dataset = AMRDataset(
                [directory],
                src_langs=[lang],
                remove_wiki=data_args.remove_wiki,
                max_samples_per_language=data_args.max_test_samples_per_language,
                is_predict=True,
            )
            predict_results = trainer.predict(
                test_dataset,
                max_length=max_length,
                num_beams=num_beams,
                penalty_alpha=training_args.penalty_alpha,
                top_k=training_args.top_k,
                do_sample=training_args.do_sample,
                top_p=training_args.top_p,
            )
            metrics = predict_results.metrics
            trainer.log_metrics(f"predict_{lang}", metrics)
            trainer.save_metrics(f"predict_{lang}", metrics)
            if trainer.is_world_process_zero():
                pf_predictions = Path(training_args.output_dir).joinpath(f"generated_predictions_{lang}.txt")
                logger.info(f"Writing predictions for {lang} to file {pf_predictions.stem}*")

                preds_linearized = tokenizer.decode_and_fix(predict_results.predictions, pbar=True)
                Path(training_args.output_dir).joinpath(f"generated_predictions_{lang}_raw.txt").write_text(
                    "\n".join(preds_linearized) + "\n", encoding="utf-8"
                )
                with pf_predictions.open("w", encoding="utf-8") as fh_preds:
                    for pred_linearized in preds_linearized:
                        try:
                            pred_penman = linearized2penmanstr(pred_linearized)
                            if AMR.parse_AMR_line(pred_penman) is None:
                                raise Exception
                            fh_preds.write(f"{pred_penman}\n")
                        except Exception:
                            fh_preds.write(f"INVALID_AMR\t{pred_linearized}\n")
                            continue


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
