import argparse
import os
import torch

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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


def main(training_mode: str = "basic") -> None:
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

    processor = AutoProcessor.from_pretrained(
        base_params.aya_processor_name, trust_remote_code=True
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

    model = AutoModelForImageTextToText.from_pretrained(
        base_params.aya_model_name,
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

    lora_config = _build_lora_config(lora_params, training_mode)
    model = get_peft_model(model, lora_config)

    #sanity checks
    # a) Print trainable params
    model.print_trainable_parameters()

    # b) Any trainable params at all?
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable > 0, "No trainable params — LoRA targets likely mismatched."

    # c) Your batch truly supervises tokens
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
    trainer.model.save_pretrained(best_model_dir)
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
    cli_args = parser.parse_args()
    main(training_mode=cli_args.training_mode)
