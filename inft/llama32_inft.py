import torch
from dataloader import build_processed_dataset 
from params import MUNIChusLoadConfig
from params import ( BasicParams, QLoRAParams, LoRAParams, MUNIChusLoadConfig, SFTParams)
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed
from llamacollator import LlamaCollator
import os
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
def main():
    bp = BasicParams()
    ql = QLoRAParams()
    lp = LoRAParams()
    cfg = MUNIChusLoadConfig(split="train")
    sp = SFTParams()

    set_seed(sp.seed)

    if not os.path.exists(bp.logs_dir):
        os.makedirs(bp.logs_dir)

    task_tag = "qlora-lora-llama32"

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
    lora_cfg = LoraConfig(
        r=lp.r,
        lora_alpha=lp.alpha,
        lora_dropout=lp.dropout,
        bias=lp.bias,
        target_modules=list(lp.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- training args ---
    args = TrainingArguments(
            output_dir= sp.llama_32_output_dir,
            # memory/throughput
            per_device_train_batch_size=1, 
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            # optimizer (QLoRA best practice)
            optim="adamw_torch_fused", #"adamw_torch",
            learning_rate=1.5e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            # precision and stability
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            torch_compile=False,
            # logging & saving
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            save_safetensors=True,
            report_to="none",
            # eval: we’ll do custom gen-eval later
            eval_strategy="no",
            # keep pixel_values; don't drop columns from our collator
            remove_unused_columns=False,
            seed=sp.seed,
            # data_seed=getattr(sp, "seed", 42),
            # multi-GPU safety (optional)
            ddp_find_unused_parameters=False,
        )

    # sft_config = SFTConfig(packing=False, max_seq_length=bp.max_length) 
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        peft_config=lora_cfg,
        data_collator=data_collator,
        tokenizer=tok,
        max_seq_length=bp.max_length,
        callbacks=[logger_cb],
    )


    trainer_stats = trainer.train()
    print("Training complete. Metrics:", trainer_stats.metrics)
    os.makedirs(sp.llama32_best_model_dir, exist_ok=True)

    trainer.model.save_pretrained(sp.llama32_best_model_dir)
    processor.save_pretrained(sp.llama32_best_model_dir)
    print(f"Saved adapters + processor to: {sp.llama32_best_model_dir}")


if __name__ == "__main__":
    main()
