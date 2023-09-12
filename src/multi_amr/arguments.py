from dataclasses import dataclass, field
from typing import List, Optional

from transformers import Seq2SeqTrainingArguments


@dataclass(frozen=False)
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/mbart-large-cc25",
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Required when you want to use PEFT with a model whose type cannot be easily derived."},
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
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={
            "help": (
                "This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from`bitsandbytes`."
            )
        },
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "This sets the computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups."
            ),
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            "help": (
                "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`."
            ),
            "choices": ["fp4", "nf4"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the code that may be downloaded alongside some models. This may be necessary to run models like Falcon who are not fully integrated in `transformers` yet."
            )
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": ("The alpha parameter for LoRA scaling")},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": ("The dropout probability for LoRA layers")},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": ("LoRA attention dimension")},
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    use_peft: bool = field(default=False, metadata={"help": "Wether to use PEFT with LoRA or not to train adapters"})
    dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for the model"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability specifically for attention in the model. (Not used by T5 models.)"},
    )
    classif_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability of classifier layers. (Only for BART-like models.)"},
    )


@dataclass(frozen=False)
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    src_langs: List[str] = field(
        metadata={
            "help": "A list of source languages that corresponds with the given indexes in"
            " your dataset. Make sure that the right language (code) is used with"
            " respect to the model that you are using. For instance, some models require specific language codes, such"
            " as mBART 'en_XX', but others require written text 'English'"
        },
    )
    preprocessed_dataset: str = field(
        metadata={
            "help": "Paths to a dataset that has already been fully processed (not"
            " collated yet). This should be a HF Dataset that has been saved"
            " to disk and can be loaded with DatasetDict.load_from_disk"
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set per given language."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set per given language."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of test examples to this "
                "value if set per given language."
            )
        },
    )
    save_amrs: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save (in)valid generated AMRs to a file 'invalid-amrs.txt' in the output directory."
                " During prediction ('--do_predict') predictions are written to 'generated_predictions_{lang}'"
                " regardless of this flag."
            )
        },
    )
    remove_wiki: bool = field(
        default=True,
        metadata={"help": ("Whether to remove the special 'wiki:' tags from the AMR.")},
    )
    use_spring_label_formatting: bool = field(
        default=False,
        metadata={"help": ("Whether to use SPRING's custom method to create decoder_input_ids automatically, which"
                           " is different from BART's default. When using this, make sure to also set"
                           " `decoder_start_token_id` correctly. For BART, using the custom spring formatting, this should then be 0")},
    )


@dataclass(frozen=False)
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
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to try to create batches of homogenous lengths (only works together with 'group_by_lang')."
        },
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
    # For generation arguments, see: https://huggingface.co/blog/how-to-generate
    do_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use sampling during generation (evaluation/prediction). Only works if"
            " predict_with_generate=True."
        },
    )
    penalty_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "The values balance the model confidence and the degeneration penalty in contrastive search"
            " decoding (evaluation/prediction). If a value is given together with 'topk', the generation will use"
            " contrastive decoding. See https://huggingface.co/blog/introducing-csearch. For generating English,"
            " the paper authors suggest penalty_alpha=0.6 and top_k=4. Only works if predict_with_generate=True."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of highest probability vocabulary.py tokens to keep for top-k sampling if do_sample=True"
            " (evaluation/prediction). If a value is given together with 'penalty_alpha', the generation will"
            " use contrastive decoding. See 'penalty_alpha' for more. Only works if predict_with_generate=True."
        },
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "The percentage of highest probability vocabulary.py tokens to keep for top-p sampling if"
            " do_sample=True (evaluation/prediction). In other words: sample from the most probable vocabulary.py"
            " items that, combined, account for p%. Only works if predict_with_generate=True."
        },
    )
    sweep_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "A YAML file containing a sweep configuration. If given, will do hyperparameter optimisation."
            " See https://docs.wandb.ai/guides/sweeps/define-sweep-configuration"
        },
    )
    use_spring_sampler: bool = field(
        default=False,
        metadata={"help": ("Whether to use the SPRING dataloader which creates batches measured in no. tokens. Update batch_size_tokens accordingly")},
    )
    batch_size_tokens: int = field(
        default=500,
        metadata={"help": ("How many tokens to use in each batch when using the SPRING dataloader when `use_spring_sampler=True`.")},
    )
