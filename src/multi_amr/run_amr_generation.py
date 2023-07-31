"""Finetuning models on multilingual datasets for text-to-AMR"""
import logging
import math
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List

import evaluate as evaluate
import numpy as np
import penman
import smatch
import transformers
from amr import AMR
from multi_amr.arguments import DataTrainingArguments, ExpandedSeq2SeqTrainingArguments, ModelArguments
from multi_amr.data.dataset import AMRDataset, collate_amr
from multi_amr.data.linearization import linearized2penmanstr
from multi_amr.data.tokenization import AMRTokenizerWrapper
from multi_amr.parse_cli import parse_cli
from multi_amr.peft_callback import PeftSavingCallback
from multi_amr.trainer import AMRTrainer
from multi_amr.utils.smart_initialization import freeze_encoder, smart_initialization
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import AutoModelForSeq2SeqLM, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = parse_cli(
        ModelArguments, DataTrainingArguments, ExpandedSeq2SeqTrainingArguments
    )

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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu},"
        + f" distributed training: {training_args.parallel_mode.value == 'distributed'},"
          f" 16-bits training: {training_args.fp16}"
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
    # Load tok_wrapper and model #
    ############################
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.tokenizer_name and model_args.model_name_or_path:
        model_args.tokenizer_name = model_args.model_name_or_path

    tok_wrapper = AMRTokenizerWrapper.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs, src_lang="en_XX")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.config_name and model_args.model_name_or_path:
        model_args.config_name = model_args.model_name_or_path

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    model.resize_token_embeddings(len(tok_wrapper.tokenizer))

    if training_args.smart_initialization:
        if Path(model_args.model_name_or_path).exists() or (training_args.do_eval and not training_args.do_train):
            logger.warning(
                "You have enabled smart initialization but you seem to be loading a model that you have"
                " already trained. This may lead to worse-than-expected performance because you will be"
                " effectively overwriting the token embeddings of the added tokens"
            )
        model = smart_initialization(model, tok_wrapper, noise_range=training_args.noise_range)

    if training_args.freeze_encoder:
        model = freeze_encoder(model)

    callbacks = []
    if model_args.use_peft:
        try:
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model.config.model_type]
        except (KeyError, AttributeError):
            if model_args.model_type is not None:
                target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_args.model_type]
            else:
                raise KeyError(
                    "Cannot automatically derive model type. Specify '--model_type' explicitly."
                    " See https://github.com/huggingface/peft/blob/e06d94ddeb6c70913593740618df76908b918d66/src/peft/utils/other.py#L262"
                )

        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        logger.info(f"Targetting {target_modules} with LoRA.")

        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, peft_config)
        callbacks.append(PeftSavingCallback)

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
                ref_penman = linearized2penmanstr(ref)

                try:
                    # First parse with `penman`, which is less sensitive than AMR.parse_AMR_line
                    # and then back to valid penman string
                    pred_penman = linearized2penmanstr(pred)
                    pred_parsed = penman.parse(pred_penman)
                    pred_penman = penman.format(pred_parsed)
                    if AMR.parse_AMR_line(pred_penman) is None:
                        raise Exception
                except Exception:
                    n_invalid += 1
                    if data_args.save_amrs:
                        fh_invalid.write(f"PRED_ERROR\nPRED: {pred_penman}\nREF: {ref_penman}\nPRED LINEAR: {pred}"
                                         f"\nREF LINEAR: {ref}\n\n")
                    continue

                try:
                    best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
                        ref_penman, pred_penman, sent_num=sentid
                    )
                except Exception:
                    n_invalid += 1
                    # At this point, any error is probably caused by the prediction
                    if data_args.save_amrs:
                        fh_invalid.write(f"SMATCH_ERROR\nPRED: {pred_penman}\nREF: {ref_penman}\nPRED LINEAR: {pred}"
                                         f"\nREF LINEAR: {ref}\n\n")
                    continue

                total_match_num += best_match_num
                total_test_num += test_triple_num
                total_gold_num += gold_triple_num
                # clear the matching triple dictionary for the next AMR pair
                smatch.match_triple_dict.clear()
                # First the prediction, then the reference AMR
                if data_args.save_amrs:
                    fh_valid.write(f"PRED: {pred_penman}\nREF: {ref_penman}\nPRED LINEAR: {pred}\nREF LINEAR: {ref}\n\n")

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
        labels_for_bleu = np.where(labels != -100, labels, tok_wrapper.tokenizer.pad_token_id)
        preds_for_bleu = np.where(preds != -100, preds, tok_wrapper.tokenizer.pad_token_id)
        ref_linearizations = tok_wrapper.decode_and_fix(labels_for_bleu)
        pred_linearizations = tok_wrapper.decode_and_fix(preds_for_bleu)
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
        tokenizer=tok_wrapper.tokenizer,
        data_collator=partial(
            collate_amr,
            tok_wrapper=tok_wrapper,
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
        trainer.save_model()  # Saves the tok_wrapper too for easy upload
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

                preds_linearized = tok_wrapper.decode_and_fix(predict_results.predictions, pbar=True)
                Path(training_args.output_dir).joinpath(f"generated_predictions_{lang}_raw.txt").write_text(
                    "\n".join(preds_linearized) + "\n", encoding="utf-8"
                )
                with pf_predictions.open("w", encoding="utf-8") as fh_preds:
                    for pred_linearized in preds_linearized:
                        try:
                            pred_penman = linearized2penmanstr(pred_linearized)
                            pred_parsed = penman.parse(pred_penman)
                            pred_penman = penman.format(pred_parsed)
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
