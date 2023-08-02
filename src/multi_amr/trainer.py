from typing import Optional

import torch
from multi_amr.data.dataset import AMRDataset
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
                    group_by_length=self.args.group_by_length,
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
                    group_by_length=self.args.group_by_length,
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

    def _get_eval_sampler(self, eval_dataset: AMRDataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_lang:
            if self.args.world_size <= 1:
                # We use batch_size * gradient_accumulation_steps as a single batch size
                # so that every optimization step, we are optimizing for a single language
                return SrcLangGroupedSampler(
                    batch_size=self.args.per_device_eval_batch_size,
                    keep_incomplete_batches=self.args.keep_incomplete_batches,
                    dataset=eval_dataset,
                    shuffle=False,
                    group_by_length=self.args.group_by_length,
                )
            else:
                return DistributedSrcLangGroupedSampler(
                    batch_size=self.args.per_device_eval_batch_size,
                    dataset=eval_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    keep_incomplete_batches=self.args.keep_incomplete_batches,
                    shuffle=False,
                    group_by_length=self.args.group_by_length,
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