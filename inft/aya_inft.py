import argparse
import os
import torch
from typing import Tuple

from dataloader import build_processed_dataset
from params import (
    BasicParams,
    LoRAParams,
    MUNIChusLoadConfig,
    QLoRAParams,
    SFTParams,
)
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, BitsAndBytesConfig
from transformers.trainer_utils import set_seed
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from ayacollator import AyaCollator
from logger import JsonlLogger, TrainerLoggingCallback
import bitsandbytes as bnb
import re
from collections import Counter

# discovers the trainable linear modules in the model
def list_linear_modules(model, pattern=r"multi_modal_projector\.linear_[12]|(q|k|v|o)_proj|gate_proj|up_proj|down_proj"):
    linear_types = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, linear_types):
            if pattern is None or re.search(pattern, name):
                names.append(name)
    # Pretty print
    print(f"# linear layers: {len(names)}")
    for n in names:
        print(n)
    # Quick summary by suffix
    suf = [n.rsplit(".", 1)[-1] for n in names]
    print("\n# suffix counts:", Counter(suf))
    return names



def _log_device_info() -> None:
    """Print a quick snapshot of the runtime hardware."""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
    try:
        print("bfloat16 supported:", torch.cuda.is_bf16_supported())
    except AttributeError:
        print("bfloat16 supported: n/a (older torch build)")


def _build_bnb_config(qlora: QLoRAParams) -> BitsAndBytesConfig:
    compute_dtype = (
        torch.bfloat16 if qlora.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=qlora.load_in_4bit,
        bnb_4bit_use_double_quant=qlora.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_lora_config(lora: LoRAParams, training_mode: str) -> LoraConfig:
    target_modules = (
        lora.aya_advanced_target_modules if training_mode == "advanced" else lora.target_modules
    )
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )


def _build_sft_args(sp: SFTParams, bp: BasicParams, output_dir: str) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=sp.gradient_accumulation_steps,
        optim="adamw_torch_fused",          # or "paged_adamw_8bit"
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=sp.warmup_ratio,
        weight_decay=sp.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=sp.max_grad_norm,
        logging_steps=sp.logging_steps,
        save_strategy="steps",
        save_steps=sp.save_steps,
        save_total_limit=sp.save_total_limit,
        save_safetensors=True,
        report_to="none",
        remove_unused_columns=False,
        seed=sp.seed,
        ddp_find_unused_parameters=False,
        packing=False,
        max_seq_length=bp.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def _resolve_model_source(
    load_from_checkpoint: bool, is_advanced: bool, sp: SFTParams, bp: BasicParams
) -> Tuple[str, bool, bool]:
    """
    Choose between a locally saved checkpoint directory and the base HF repo.
    Returns (model_path, use_checkpoint_processor, has_adapter_weights).
    """
    if load_from_checkpoint:
        candidate = sp.aya_adv_best_model_dir if is_advanced else sp.aya_best_model_dir
        if os.path.isdir(candidate) and os.listdir(candidate):
            print(f"Loading model + processor from checkpoint: {candidate}")
            return candidate, True, False
        print(f"Checkpoint path '{candidate}' missing or empty; searching for latest checkpoint directory.")
        output_dir = sp.aya_adv_output_dir if is_advanced else sp.aya_output_dir
        if os.path.isdir(output_dir):
            candidates = [os.path.join(output_dir, name) for name in os.listdir(output_dir)]
            dir_candidates = [path for path in candidates if os.path.isdir(path)]
            pattern = re.compile(r"checkpoint[-_](\d+)$")
            numbered = []
            fallback = []
            for path in dir_candidates:
                match = pattern.search(os.path.basename(path))
                if match:
                    numbered.append((int(match.group(1)), path))
                else:
                    fallback.append(path)
            if numbered:
                best_path = max(numbered, key=lambda item: item[0])[1]
                print(f"Found checkpoint directory: {best_path}")
                has_adapter = os.path.isfile(os.path.join(best_path, "adapter_config.json"))
                return best_path, False, has_adapter
            if fallback:
                best_path = max(fallback, key=os.path.getmtime)
                print(f"Using most recent directory under output_dir: {best_path}")
                has_adapter = os.path.isfile(os.path.join(best_path, "adapter_config.json"))
                return best_path, False, has_adapter
        print("No suitable checkpoint directories found; falling back to HF base model.")
    return bp.aya_model_name, False, False


def main(training_mode: str = "basic", load_from_checkpoint: bool = False) -> None:
    training_mode = training_mode.lower()
    if training_mode not in {"basic", "advanced"}:
        raise ValueError("training_mode must be either 'basic' or 'advanced'")

    _log_device_info()

    base_params = BasicParams()
    qlora_params = QLoRAParams()
    lora_params = LoRAParams()
    sft_params = SFTParams()
    data_cfg = MUNIChusLoadConfig(split="train")
    is_advanced = training_mode == "advanced"

    set_seed(sft_params.seed)

    os.makedirs(base_params.logs_dir, exist_ok=True)
    os.makedirs(base_params.output_dir, exist_ok=True)

    task_tag = "qlora-lora-aya8b-advanced" if is_advanced else "qlora-lora-aya8b"
    log_path = os.path.join(base_params.logs_dir, f"trainer_{task_tag}.jsonl")
    json_logger = JsonlLogger(path=log_path)
    trainer_cb = TrainerLoggingCallback(json_logger, run_name=f"inft-{task_tag}-trainer")

    model_source, use_checkpoint_processor, has_adapters = _resolve_model_source(
        load_from_checkpoint=load_from_checkpoint,
        is_advanced=is_advanced,
        sp=sft_params,
        bp=base_params,
    )
    processor_source = model_source if use_checkpoint_processor else base_params.aya_processor_name

    processor = AutoProcessor.from_pretrained(
        processor_source, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = AyaCollator(
        processor=processor,
        max_length=base_params.max_length,
        input_data_format="channels_last",
        add_generation_prompt=False,
    )

    print("Building processed dataset...")
    dataset = build_processed_dataset(cfg=data_cfg)
    print(f"Train split size: {len(dataset)} examples")

    quant_config = _build_bnb_config(qlora_params)

    ##############################################################################################
    # --- model ---
    # loads the basemode fom HF and combines the QLoRA adapters 
    if load_from_checkpoint and has_adapters:
        print(f"loading base model from:{base_params.aya_model_name} and adapters from{model_source}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_params.aya_model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=base_params.attn_impl,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        # load lora adapters 
        model = PeftModel.from_pretrained(base_model, model_source)
    else:
        # full HF style mode from local dir i.e best_model dir, and prepares for PEFT 
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=base_params.attn_impl,
        )
        
        # if is_advanced:
        #     print("Advanced training mode: Finetubeable Linear modules")
        #     print("\n".join(list_linear_modules(model)))

        model = prepare_model_for_kbit_training(model)

    if hasattr(model, "config"):
        model.config.use_cache = False

    # applting lora_config if it is loaded form best_model or HF 
    if not (load_from_checkpoint and has_adapters):
        lora_config = _build_lora_config(lora_params, training_mode)
        model = get_peft_model(model, lora_config)

    #sanity checks
    # Print trainable params
    model.print_trainable_parameters()

    # Any trainable params at all?
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable > 0, "No trainable params — LoRA targets likely mismatched."

    # batch truly supervises tokens
    from torch.utils.data import DataLoader
    b = next(iter(DataLoader(dataset, batch_size=data_cfg.batch_size, collate_fn=data_collator)))
    print("supervised tokens per row:", (b["labels"] != -100).sum(dim=1))
    assert (b["labels"] != -100).any(), "All labels are -100 — no loss will flow."

    output_dir = sft_params.aya_adv_output_dir if is_advanced else sft_params.aya_output_dir
    os.makedirs(output_dir, exist_ok=True)
    sft_args = _build_sft_args(sft_params, base_params, output_dir)

    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[trainer_cb], 
    )

    print("Starting supervised fine-tuning...")
    train_result = trainer.train()
    print("Training complete. Metrics:", train_result.metrics)

    best_model_dir = (
        sft_params.aya_adv_best_model_dir if is_advanced else sft_params.aya_best_model_dir
    )
    os.makedirs(best_model_dir, exist_ok=True)
    print("Merging LoRA adapters into the base model (bf16) before saving...")
    merged_model = trainer.model.merge_and_unload()
    merged_model = merged_model.to(dtype=torch.bfloat16)
    merged_model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"Saved adapters + processor to: {best_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Aya 8B vision model.")
    parser.add_argument(
        "--training-mode",
        choices=["basic", "advanced"],
        default="basic",
        help="Choose between the basic LoRA recipe or an advanced variant.",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        action="store_true",
        help="If set, initialize weights/processors from the latest best_model_dir or checkpoint instead of the HF base model.",
    )
    cli_args = parser.parse_args()
    main(training_mode=cli_args.training_mode, load_from_checkpoint=cli_args.load_from_checkpoint)
