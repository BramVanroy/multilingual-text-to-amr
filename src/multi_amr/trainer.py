from typing import Optional

import torch
from datasets import Dataset
from multi_amr.data.sampler import DistributedSrcLangGroupedSampler, SrcLangGroupedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Seq2SeqTrainer
from transformers.trainer_pt_utils import DistributedSamplerWithLoop, ShardSampler
from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode


class AMRTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_lang:
            # We use batch_size * gradient_accumulation_steps as a single batch size
            # so that every optimization step, we are optimizing for a single language
            return SrcLangGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                keep_incomplete_batches=self.args.keep_incomplete_batches,
                dataset=self.train_dataset,
                shuffle=self.args.shuffle,
                group_by_length=self.args.group_by_length,
            )
        else:
            return RandomSampler(self.train_dataset)

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_lang:
            # We use batch_size * gradient_accumulation_steps as a single batch size
            # so that every optimization step, we are optimizing for a single language
            return SrcLangGroupedSampler(
                batch_size=self.args.per_device_eval_batch_size,
                keep_incomplete_batches=True,
                dataset=eval_dataset,
                shuffle=False,
                group_by_length=False,
            )
        else:
            return SequentialSampler(eval_dataset)
