from dataclasses import dataclass, field
from functools import partialmethod
from typing import Dict, List, Optional

import torch
from mbart_amr.data.sampler import (DistributedSrcLangGroupedSampler,
                                    SrcLangGroupedSampler)
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_pt_utils import (DistributedSamplerWithLoop,
                                           ShardSampler)
from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode


@dataclass
class ExpandedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Stop training when the evaluation metric worsens (instead of improves) for"
            " early_stopping_patience evaluation calls."
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Denote how much the evaluation metric must improve to satisfy early stopping conditions."},
    )
    group_by_lang: bool = field(
        default=True,
        metadata={"help": "Whether to try to create batches of homogenous languages."},
    )
    keep_incomplete_batches: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep 'rest' batches at the end that can contain samples of different languages."
        },
    )
    shuffle: bool = field(
        default=True,
        metadata={
            "help": "Whether to shuffle the training set when 'keep_incomplete_batches' is enabled. If"
            " 'keep_incomplete_batches' is not enabled, the training set will always be shuffled."
            " The validation/test set will never be shuffled."
        },
    )
    smart_initialization: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize the embeddings of the newly added tokens in a 'smart' way based on their"
            " semantics."
        },
    )
    noise_range: float = field(
        default=0.1,
        metadata={
            "help": "The amount of noise to add during smart initialization to the tokens that are similar to other"
                    " tokens. Noise is generated from a uniform distribution that spans [-noise_range, +noise_range]."
                    " The default is the default noise used in SPRING"
        },
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the encoder and only train the decoder. The shared embeddings will not be frozen"
        },
    )
    penalty_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "The values balance the model confidence and the degeneration penalty in contrastive search"
            " decoding. If a value is given together with 'topk', the generation will use contrastive"
            " decoding. See https://huggingface.co/blog/introducing-csearch. For generating English, the paper"
            " authors suggest penalty_alpha=0.6 and top_k=4. This only works if 'predict_with_generate' is"
            " enabled. Note that this will make evaluation very, very slow! Best to keep that just for"
            " predicting the test set."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering. If a value is"
            " given together with 'penalty_alpha', the generation will use contrastive"
            " decoding. See 'penalty_alpha' for more."
        },
    )


class AMRTrainer(Seq2SeqTrainer):
    def __init__(self, *args, penalty_alpha: Optional[float] = None, top_k: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._penalty_alpha = penalty_alpha
        self._top_k = top_k

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        gen_kwargs = {**gen_kwargs, "penalty_alpha": self._penalty_alpha, "top_k": self._top_k}
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    def predict(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        gen_kwargs = {**gen_kwargs, "penalty_alpha": self._penalty_alpha, "top_k": self._top_k}
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_lang:
            if self.args.world_size <= 1:
                # We use batch_size * gradient_accumulation_steps as a single batch size
                # so that every optimization step, we are optimizing for a single language
                return SrcLangGroupedSampler(
                    batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    keep_incomplete_batches=self.args.keep_incomplete_batches,
                    dataset=self.train_dataset,
                    shuffle=self.args.shuffle,
                    generator=generator,
                )
            else:
                return DistributedSrcLangGroupedSampler(
                    batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    keep_incomplete_batches=self.args.keep_incomplete_batches,
                    shuffle=self.args.shuffle,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_lang:
            if self.args.world_size <= 1:
                # We use batch_size * gradient_accumulation_steps as a single batch size
                # so that every optimization step, we are optimizing for a single language
                return SrcLangGroupedSampler(
                    batch_size=self.args.per_device_eval_batch_size,
                    keep_incomplete_batches=self.args.keep_incomplete_batches,
                    dataset=eval_dataset,
                    shuffle=False,
                )
            else:
                return DistributedSrcLangGroupedSampler(
                    batch_size=self.args.per_device_eval_batch_size,
                    dataset=eval_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    keep_incomplete_batches=self.args.keep_incomplete_batches,
                    shuffle=False,
                )

        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )
