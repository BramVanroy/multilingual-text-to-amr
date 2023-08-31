"""Finetuning models on multilingual datasets for text-to-AMR"""
import dataclasses
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
import yaml

from datasets import Dataset, DatasetDict
from multi_amr.arguments import DataTrainingArguments, ExpandedSeq2SeqTrainingArguments, ModelArguments
from multi_amr.data.collator import collate_amr
from multi_amr.data.postprocessing_graph import ParsedStatus, tokens2graph
from multi_amr.data.postprocessing_str import postprocess_str_after_delinearization, tokenize_except_quotes_and_angles
from multi_amr.data.tokenization import AMRTokenizerWrapper, TokenizerType
from multi_amr.parse_cli import parse_cli
from multi_amr.peft_callback import PeftSavingCallback
from multi_amr.trainer import AMRTrainer
from multi_amr.utils.smart_initialization import freeze_encoder, smart_initialization
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from smatchpp import Smatchpp, preprocess, solvers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, EarlyStoppingCallback, set_seed, AutoConfig
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = parse_cli(
        ModelArguments, DataTrainingArguments, ExpandedSeq2SeqTrainingArguments
    )
    training_args = dataclasses.replace(training_args, remove_unused_columns=False)

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

    max_length = tok_wrapper.tokenizer.model_max_length

    if max_length > 2048:  # Some max lengths are set to LARGE INT
        if tok_wrapper.tokenizer_type == TokenizerType.BLOOM:
            # Taken from their paper
            max_length = 2048
        elif tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.BART):
            # mbart-large-cc25 is set to 1024 but mbart-large-50-may-to-one-mmt is set to LARGE INT
            max_length = 1024
        elif tok_wrapper.tokenizer_type == TokenizerType.T5:
            # 1024 according to the mT5 paper
            max_length = 1024

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if not model_args.config_name and model_args.model_name_or_path:
        model_args.config_name = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)

    if tok_wrapper.tokenizer_type in (TokenizerType.BART, TokenizerType.MBART, TokenizerType.NLLB):
        config.dropout = model_args.dropout
    elif tok_wrapper.tokenizer_type in (TokenizerType.T5,):
        config.dropout_rate = model_args.dropout
    elif tok_wrapper.tokenizer_type in (TokenizerType.BLOOM,):
        config.hidden_dropout = model_args.dropout

    with training_args.main_process_first(desc="(Down)loading model"):
        if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.BART, TokenizerType.T5):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tok_wrapper.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tok_wrapper.tokenizer))
            config.vocab_size = len(tok_wrapper.tokenizer)

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

    ####################
    # CALCULATE SMATCH #
    ####################
    graph_standardizer = preprocess.AMRStandardizer()
    # Using hillclimber here. Not the best accuracy but ILP is causing some issues, which are very disrupting for
    # training the models. https://github.com/flipz357/smatchpp/issues/4
    # So during evaluation (separate script) we'll make use of ILP instead
    # ilp = solvers.ILP()
    solver = solvers.HillClimber()
    smatch_metric = Smatchpp(alignmentsolver=solver, graph_standardizer=graph_standardizer)

    def calculate_smatch(references: List[str], predictions: List[str]):
        # NOTE: it is possible that on a local install, I made changes to smatchpp so that it will
        # ignore malformed pairs. This part here is therefore only applicable for training and you cannot
        # be sure to use it for prediction because it will just ignore the invalid graphs
        filtered_refs = []
        filtered_preds = []
        for ref, pred in zip(references, predictions):
            try:
                penman.decode(ref)
                penman.decode(pred)
            except:
                continue
            else:
                filtered_refs.append(ref)
                filtered_preds.append(pred)

        score, optimization_status = smatch_metric.score_corpus(filtered_refs, filtered_preds)

        score = score["main"]
        return {
            "smatch_precision": score["Precision"]["result"],
            "smatch_recall": score["Recall"]["result"],
            "smatch_fscore": score["F1"]["result"],
            "smatch_unparsable": len(references) - len(filtered_refs)
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
        preds_penman, preds_status = preds_decoded["penman"], preds_decoded["status"]

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
        src_langs_idxs = _dataset.unique("src_lang_idx")
        for lang_idx in src_langs_idxs:
            try:
                logger.info(f"Setting lang index {lang_idx} to {data_args.src_langs[lang_idx]}")
            except IndexError:
                raise IndexError(
                    f"It seems that you have more indices in your {split_type} dataset ({src_langs_idxs})"
                    f" than you have specified 'src_langs' ({data_args.src_langs}) in your script config"
                    f" or arguments."
                )

    train_dataset = None
    validation_dataset = None
    test_dataset = None
    raw_datasets = DatasetDict.load_from_disk(data_args.preprocessed_dataset)
    logger.info(f"Using preprocessed datasets at {data_args.preprocessed_dataset}")
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise KeyError("--do_train requires a train dataset")

        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        check_lang_idx(train_dataset, "train")

    if training_args.do_train or training_args.do_eval:
        if "validation" not in raw_datasets:
            raise KeyError("--do_eval and --do_train require a validation dataset")

        validation_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(validation_dataset), data_args.max_eval_samples)
            validation_dataset = validation_dataset.select(range(max_eval_samples))
        check_lang_idx(validation_dataset, "validation")

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise KeyError("--do_predict requires a test dataset")

        test_dataset = raw_datasets["test"]
        if data_args.max_test_samples is not None:
            max_test_samples = min(len(test_dataset), data_args.max_test_samples)
            test_dataset = test_dataset.select(range(max_test_samples))
        check_lang_idx(test_dataset, "test")

    logger.info(f"Loaded datasets!")
    if train_dataset:
        logger.info(f"Train: {str(train_dataset)}")
    if validation_dataset:
        logger.info(f"Validation: {str(validation_dataset)}")
    if test_dataset:
        logger.info(f"Test: {str(test_dataset)}")

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

    def model_init(trial):
        # Sizes will be mismatched here because config already includes added tokens but the original checkpoint does not
        if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.BART, TokenizerType.T5):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tok_wrapper.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tok_wrapper.tokenizer))
            config.vocab_size = len(tok_wrapper.tokenizer)

        if training_args.smart_initialization:
            model = smart_initialization(model, tok_wrapper, noise_range=training_args.noise_range)

        return model

    # Initialize our Trainer
    trainer = AMRTrainer(
        model=None if training_args.sweep_config else model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tok_wrapper.tokenizer,
        data_collator=partial(
            collate_amr,
            src_langs=data_args.src_langs,
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
        model_init=model_init if training_args.sweep_config else None,
    )

    if training_args.sweep_config:
        def wandb_hp_space(trial=None):
            return yaml.safe_load(Path(training_args.sweep_config).read_text(encoding="utf-8"))
        best_trial = trainer.hyperparameter_search(
            backend="wandb",
            metric="eval/smatch_fscore",
            hp_space=wandb_hp_space,
            n_trials=wandb_hp_space()["run_cap"] if "run_cap" in wandb_hp_space() else None,
            direction="maximize",
        )

        logging.info(f"Best hyperparameter search run: {best_trial.run_id}")
        with Path(training_args.output_dir).joinpath("wandb_best_hparams.json").open("w", encoding="utf-8") as hp_out:
            best_trial.hyperparameters.pop("assignments", None)
            best_trial.hyperparameters["metric"] = "eval/smatch_fscore"
            hparams_dump = {
                **best_trial.hyperparameters,
                "best_run": best_trial.run_id,
                "objective": best_trial.objective
            }
            dump(hparams_dump, hp_out, indent=4, sort_keys=True)

        for hparam, v in best_trial.hyperparameters.items():
            setattr(trainer.args, hparam, v)

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
    #
    # if trainer.is_world_process_zero():
    #     for split in ("valid", "test"):
    #         if split == "valid":
    #             if not training_args.do_eval:
    #                 continue
    #             dataset = validation_dataset
    #         else:
    #             if not training_args.do_predict:
    #                 continue
    #             dataset = test_dataset
    #
    #         logger.info(f"*** {split.upper()} ***")
    #         df = dataset.to_pandas()
    #         df = df.drop(columns=["__index_level_0__"], errors="ignore")
    #
    #         for src_lang_idx, group_ds in df.groupby("src_lang_idx"):
    #             src_lang = data_args.src_langs[src_lang_idx]
    #             lang_dataset = Dataset.from_pandas(group_ds)
    #             predict_results = trainer.predict(
    #                 lang_dataset,
    #                 max_length=training_args.generation_max_length,
    #                 num_beams=training_args.num_beams,
    #                 penalty_alpha=training_args.penalty_alpha,
    #                 top_k=training_args.top_k,
    #                 do_sample=training_args.do_sample,
    #                 top_p=training_args.top_p,
    #                 metric_key_prefix=f"{split}_{src_lang}",
    #             )
    #             metrics = predict_results.metrics
    #             trainer.log_metrics(f"{split}_{src_lang}", metrics)
    #             trainer.save_metrics(f"{split}_{src_lang}", metrics)
    #             # TODO: just write custom `predict` script?
    #             preds = predict_results.predictions
    #             labels = predict_results.label_ids
    #
    #             num_samples = len(labels)
    #
    #             # CLM models need to reshift their labels
    #             if tok_wrapper.tokenizer_type in (TokenizerType.BLOOM,):
    #                 preds = preds[:, :-1]
    #
    #             # SMATCH
    #             labels_for_smatch = np.where(labels != -100, labels, tok_wrapper.tokenizer.pad_token_id)
    #             preds_for_smatch = np.where(preds != -100, preds, tok_wrapper.tokenizer.pad_token_id)
    #             refs_penman = tok_wrapper.batch_decode_amr_ids(labels_for_smatch, reset_variables=True)["penman"]
    #             preds_decoded = tok_wrapper.batch_decode_amr_ids(preds_for_smatch, reset_variables=True)
    #             preds_penman, preds_status = preds_decoded["penman"], preds_decoded["status"]
    #
    #             pf_predictions = Path(training_args.output_dir).joinpath(f"{split}_predictions_{src_lang}.txt")
    #             logger.info(f"Writing {split} predictions for {src_lang} to file {pf_predictions.stem}*")
    #
    #             with pf_predictions.open("w", encoding="utf-8") as fhout:
    #                 for sent, ref, pred in zip(lang_dataset["sentence"], refs_penman, preds_penman):
    #                     fhout.write(f"{sent}\nREF: {ref}\nPRED: {pred}\n\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


# Train
# OMP_NUM_THREADS=8 WANDB_PROJECT="amr-en-cleaned-flan-t5-large" torchrun --standalone --nproc_per_node 4 src/multi_amr/run_amr_generation.py train_config_ampere.json --model_name_or_path google/flan-t5-large --output_dir results/amr30/full_cleaned/flan-t5-large/1e-5lr+50ep+64tbs --run_name 1e-5lr+50ep+64tbs --learning_rate 1e-5 |& tee output.log
# Eval
# CUDA_VISIBLE_DEVICES=0 python src/multi_amr/run_amr_generation.py predict_config_ampere.json |& tee output.log

# TODO: predict script, hparam sweep, check data lengths and filter data based on input_max_seq_length BEFORE sample/collating with dataset.filter
