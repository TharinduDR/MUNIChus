import argparse
import os

import torch
from dataloader import build_processed_dataset 
from params import ( BasicParams, QLoRAParams, LoRAParams, MUNIChusLoadConfig, SFTParams)
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed
from llamacollator import LlamaCollator
from trl import SFTTrainer, SFTConfig
from logger import JsonlLogger, TrainerLoggingCallback

# ---- CUDA INFO -----------------------------
print("CUDA avail:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Torch CUDA:", torch.version.cuda, "torch:", torch.__version__)


# Check if bfloat16 is supported
print("BF16 supported?", torch.cuda.is_bf16_supported())
print("FP16 supported?", torch.cuda.is_available())  # FP16 works if you have CUDA

# See what dtype your device prefers
print("Preferred dtype:", torch.cuda.get_device_capability())




# -----------------------------
# Training entry
# -----------------------------
def main(training_mode: str = "basic") -> None:
    training_mode = training_mode.lower()
    if training_mode not in {"basic", "advanced"}:
        raise ValueError("training_mode must be either 'basic' or 'advanced'")

    bp = BasicParams()
    ql = QLoRAParams()
    lp = LoRAParams()
    cfg = MUNIChusLoadConfig(split="train")
    sp = SFTParams()
    is_advanced = training_mode == "advanced"

    set_seed(sp.seed)

    if not os.path.exists(bp.logs_dir):
        os.makedirs(bp.logs_dir)

    task_tag = "qlora-lora-llama32-advanced" if is_advanced else "qlora-lora-llama32"

    log_path = os.path.join(bp.logs_dir, f"trainer_{task_tag}.jsonl")

    jsonloger = JsonlLogger(path=log_path)
    logger_cb = TrainerLoggingCallback(jsonloger, run_name=f"inft-{task_tag}-trainer")

    # --- processor ---
    processor = AutoProcessor.from_pretrained(bp.llama_processor_name, trust_remote_code=True)
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- train data ---
    train_ds = build_processed_dataset(cfg=cfg)
    print(f"Train size: {len(train_ds)}")

    # --- collator (training mode) ---
    data_collator = LlamaCollator(
        processor=processor,
        max_length=bp.max_length,
        input_data_format="channels_last",
        add_generation_prompt=False,
    )

    # --- 4-bit config ---
    compute_dtype = torch.bfloat16 if ql.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=ql.load_in_4bit,
        bnb_4bit_use_double_quant=ql.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=ql.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # --- model ---
    model = AutoModelForImageTextToText.from_pretrained(
        bp.llama_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=bp.attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    target_modules = lp.advanced_target_modules if is_advanced else lp.target_modules
    lora_cfg = LoraConfig(
        r=lp.r,
        lora_alpha=lp.alpha,
        lora_dropout=lp.dropout,
        bias=lp.bias,
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    
    #sanity checks
    # a) Print trainable params
    model.print_trainable_parameters()

    # b) Any trainable params at all?
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable > 0, "No trainable params — LoRA targets likely mismatched."

    # c) Your batch truly supervises tokens
    from torch.utils.data import DataLoader
    b = next(iter(DataLoader(train_ds, batch_size=cfg.batch_size, collate_fn=data_collator)))
    print("supervised tokens per row:", (b["labels"] != -100).sum(dim=1))
    assert (b["labels"] != -100).any(), "All labels are -100 — no loss will flow."



    model.config.use_cache = False

    # --- training args ---
    output_dir = sp.llama32_adv_output_dir if is_advanced else sp.llama_32_output_dir
    args = SFTConfig(
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
        max_seq_length=bp.max_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tok,
        callbacks=[logger_cb],
    )


    trainer_stats = trainer.train()
    print("Training complete. Metrics:", trainer_stats.metrics)
    best_model_dir = sp.llama32_adv_best_model_dir if is_advanced else sp.llama32_best_model_dir
    os.makedirs(best_model_dir, exist_ok=True)

    trainer.model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"Saved adapters + processor to: {best_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Llama 3.2 vision model.")
    parser.add_argument(
        "--training-mode",
        choices=["basic", "advanced"],
        default="basic",
        help="Choose between the basic LoRA recipe or an advanced variant.",
    )
    cli_args = parser.parse_args()
    main(training_mode=cli_args.training_mode)
