import importlib
import logging
from typing import Any, Optional, Tuple

import torch
import transformers
from datasets import Dataset
from multi_amr.data.sampler import SrcLangGroupedSampler, SpringSampler
from packaging import version
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import Seq2SeqTrainer
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.optimization import Adafactor
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption, has_length
from transformers.training_args import OptimizerNames, TrainingArguments
from transformers.utils import is_bitsandbytes_available, is_sagemaker_mp_enabled, strtobool

from torch.optim import RAdam

from multi_amr.data.tokenization import AMRTokenizerWrapper
import datasets

from transformers import is_datasets_available
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

logger = logging.getLogger(__name__)


class AMRTrainer(Seq2SeqTrainer):
    def __init__(self, tok_wrapper: AMRTokenizerWrapper, *trainer_args, **kwargs):
        self.tok_wrapper = tok_wrapper
        super().__init__(*trainer_args, **kwargs)

    def get_train_dataloader(self):
        if self.args.use_spring_sampler:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            # SPRING always shuffles when training (not when evaluating)
            batch_sampler = SpringSampler(train_dataset, self.tok_wrapper, self.args.batch_size_tokens, shuffle=True)
            dataloader_params = {
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "batch_sampler": batch_sampler
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["worker_init_fn"] = seed_worker

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.args.use_spring_sampler:
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            data_collator = self.data_collator

            if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

            batch_sampler = SpringSampler(eval_dataset, self.tok_wrapper, self.args.batch_size_tokens, shuffle=False)
            dataloader_params = {
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "batch_sampler": batch_sampler
            }

            return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
        else:
            return super().get_eval_dataloader()

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
            return SrcLangGroupedSampler(
                batch_size=self.args.per_device_eval_batch_size,
                keep_incomplete_batches=True,
                dataset=eval_dataset,
                shuffle=False,
                group_by_length=False,
            )
        else:
            return SequentialSampler(eval_dataset)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = AMRTrainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                if optimizer_cls == RAdam:
                    # To mimick SPRING, which does not make a distinction for wd between parameters
                    self.optimizer = RAdam(opt_model.parameters(), **optimizer_kwargs, weight_decay=self.args.weight_decay)
                else:
                    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from transformers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == "radam":  # Added RAdam dynamically in main run script
            optimizer_cls = RAdam
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim in [
            OptimizerNames.ADAMW_BNB,
            OptimizerNames.ADAMW_8BIT,
            OptimizerNames.PAGED_ADAMW,
            OptimizerNames.PAGED_ADAMW_8BIT,
            OptimizerNames.LION,
            OptimizerNames.LION_8BIT,
            OptimizerNames.PAGED_LION,
            OptimizerNames.PAGED_LION_8BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamW, Lion

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamW
                elif "lion" in args.optim:
                    optimizer_cls = Lion
                    additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}

                bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            if is_bitsandbytes_available() and version.parse(
                importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")

        return optimizer_cls, optimizer_kwargs
