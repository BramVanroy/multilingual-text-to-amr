"""Finetuning models on multilingual datasets for text-to-AMR"""
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import List

import evaluate as evaluate
import numpy as np
import penman
import transformers
from amr import AMR
from datasets import DatasetDict
from multi_amr.arguments import DataTrainingArguments, ExpandedSeq2SeqTrainingArguments, ModelArguments
from multi_amr.data.collator import collate_amr
from multi_amr.data.postprocessing_graph import ParsedStatus
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from multi_amr.parse_cli import parse_cli
from multi_amr.peft_callback import PeftSavingCallback
from multi_amr.trainer import AMRTrainer
from multi_amr.utils.smart_initialization import freeze_encoder, smart_initialization
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from smatchpp import Smatchpp, preprocess, solvers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, EarlyStoppingCallback, set_seed
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

    ##############################
    # Load tok_wrapper and model #
    ##############################
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.tokenizer_name and model_args.model_name_or_path:
        model_args.tokenizer_name = model_args.model_name_or_path

    tok_wrapper = AMRTokenizerWrapper.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs, src_lang="en_XX")

    # Check if all src_langs are supported
    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB):
        for src_lang in data_args.src_langs:
            if src_lang not in tok_wrapper.tokenizer.lang_code_to_id:
                raise KeyError(
                    f"src_lang {src_lang} not supported by this tokenizer of type"
                    f" {tok_wrapper.tokenizer_type}. Valid src_langs are"
                    f" {', '.join(tok_wrapper.tokenizer.lang_code_to_id.keys())}"
                )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.config_name and model_args.model_name_or_path:
        model_args.config_name = model_args.model_name_or_path

    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.T5):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **config_kwargs)

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
                    "Cannot automatically derive model type for LoRA. Specify '--model_type' explicitly."
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
        logger.info("Peft with LoRA enabled!")

    #######################
    # CUSTOM METRICS #
    #######################
    graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
    ilp = solvers.ILP()
    smatch_metric = Smatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer)

    def calculate_smatch(references: List[str], predictions: List[str]):
        score, optimization_status = smatch_metric.score_corpus(references, predictions)
        score = score["main"]
        return {
            "smatch_precision": score["Precision"]["result"],
            "smatch_recall": score["Recall"]["result"],
            "smatch_fscore": score["F1"]["result"],
        }

    acc_metric = evaluate.load("accuracy")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        num_samples = len(labels)
        # CLM models need to reshift their labels
        if tok_wrapper.tokenizer_type in (TokenizerType.BLOOM,):
            labels = labels[:, 1:]
            preds = preds[:, :-1]

        # SMATCH
        labels_for_smatch = np.where(labels != -100, labels, tok_wrapper.tokenizer.pad_token_id)
        preds_for_smatch = np.where(preds != -100, preds, tok_wrapper.tokenizer.pad_token_id)
        refs_penman = tok_wrapper.batch_decode_amr_ids(labels_for_smatch, reset_variables=True)["penman"]
        preds_decoded = tok_wrapper.batch_decode_amr_ids(preds_for_smatch, reset_variables=True)
        preds_penman, preds_status = preds_decoded["penman"], preds_decoded["preds_status"]

        num_not_recoverable = sum([1 for status in preds_status if status == ParsedStatus.BACKOFF])
        percent_not_recoverable = {"percent_not_recoverable": num_not_recoverable * 100 / num_samples}
        smatch_score = calculate_smatch(refs_penman, preds_penman)

        # We can only calculate accuracy when we have the same number of predicted tokens and reference tokens
        # which is only the case when predict_with_generate is false
        if not training_args.predict_with_generate:
            # Accuracy: flatten and calculate accuracy on flattened arrays
            labels_for_acc = labels.reshape(-1)
            preds_for_acc = preds.reshape(-1)
            mask = labels_for_acc != -100
            labels_for_acc = labels_for_acc[mask]
            preds_for_acc = preds_for_acc[mask]
            acc = acc_metric.compute(predictions=preds_for_acc, references=labels_for_acc)
            return {**acc, **smatch_score, **percent_not_recoverable}

        return {**smatch_score, **percent_not_recoverable}

    #######################
    # Load datasets #
    #######################
    def check_lang_idx(_dataset: Dataset, split_type: str):
        src_langs_idxs = set(_dataset["src_lang_idx"].to_list())
        for lang_idx in src_langs_idxs:
            try:
                logger.info(f"Setting lang index {lang_idx} to {data_args.src_langs[lang_idx]}")
            except IndexError:
                raise IndexError(f"It seems that you have more indices in your {split_type} dataset ({src_langs_idxs})"
                                 f" than you have specified 'src_langs' ({data_args.src_langs}) in your script config"
                                 f" or arguments.")

    train_dataset = None
    validation_dataset = None
    raw_datasets = DatasetDict.load_from_disk(data_args.preprocessed_dataset)
    logger.info(f"Using preprocessed datasets at {data_args.preprocessed_dataset}")
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        check_lang_idx(train_dataset, "train")

    if training_args.do_eval or training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval and --do_train require a validation dataset")

        validation_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(validation_dataset), data_args.max_eval_samples)
            validation_dataset = validation_dataset.select(range(max_eval_samples))
        check_lang_idx(train_dataset, "validation")

    logger.info(f"Loaded datasets!")
    if train_dataset:
            logger.info(f"Train: {str(train_dataset)}")
    if validation_dataset:
        logger.info(f"Validation: {str(validation_dataset)}")

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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
            penalty_alpha=training_args.penalty_alpha,
            top_k=training_args.top_k,
            do_sample=training_args.do_sample,
            top_p=training_args.top_p,
        )

        # Will take max_eval_samples into consideration
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
        raise    NotImplementedError("do_predict not implemented yet")
        # TODO: implement this for --preprocessed_dataset
        # Loop over every language separately to avoid missing batches from the dataloader
        # and to keep track of the performance of each language separately
        for lang, directory in zip(data_args.src_langs, data_args.test_directories):
            test_dataset = AMRDataset(
                [directory],
                src_langs=[lang],
                remove_wiki=data_args.remove_wiki,
                max_samples_per_language=max(1, data_args.max_test_samples // len(data_args.src_langs)),
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

                preds_linearized = tok_wrapper.decode_and_fix_amr(predict_results.predictions, pbar=True)
                Path(training_args.output_dir).joinpath(f"generated_predictions_{lang}_raw.txt").write_text(
                    "\n".join(preds_linearized) + "\n", encoding="utf-8"
                )
                with pf_predictions.open("w", encoding="utf-8") as fh_preds:
                    for pred_linearized in preds_linearized:
                        try:
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
