import argparse
import os
import re
from typing import Tuple
from utils import merge_lora_adapter
import torch
from dataloader import build_processed_dataset 
from params import ( BasicParams, QLoRAParams, LoRAParams, MUNIChusLoadConfig, SFTParams)
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
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


def _resolve_model_source(
    load_from_checkpoint: bool, is_advanced: bool, sp: SFTParams, bp: BasicParams
) -> Tuple[str, bool, bool]:
    """
    Decide whether to initialize training from an existing checkpoint directory or the base HF repo.
    Returns (model_path, use_checkpoint_processor, has_adapter_weights).
    """
    if load_from_checkpoint:
        candidate = sp.llama32_adv_best_model_dir if is_advanced else sp.llama32_best_model_dir
        if os.path.isdir(candidate) and os.listdir(candidate):
            print(f"Loading model + processor from checkpoint: {candidate}")
            return candidate, True, False
        print(f"Checkpoint path '{candidate}' missing or empty; searching for latest checkpoint directory.")
        output_dir = sp.llama32_adv_output_dir if is_advanced else sp.llama_32_output_dir
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
    return bp.llama_model_name, False, False



# Training 
def main(training_mode: str = "basic", load_from_checkpoint: bool = False) -> None:
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

    # model source is local lora-adapters if HF style best_model does not exists
    model_source, use_checkpoint_processor, has_adapters = _resolve_model_source(
        load_from_checkpoint=load_from_checkpoint, is_advanced=is_advanced, sp=sp, bp=bp
    )
    processor_source = model_source if use_checkpoint_processor else bp.llama_processor_name

    # --- processor ---
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
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

    ##############################################################################################
    # --- model ---
    # loads the basemode fom HF and combines the QLoRA adapters 
    if load_from_checkpoint and has_adapters:
        print(f"loading base model from:{bp.llama_model_name} and adapters from{model_source}")
        # load HF model
        base_model = AutoModelForImageTextToText.from_pretrained(
            bp.llama_model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=bp.attn_impl,
        )    
        base_model = prepare_model_for_kbit_training(base_model)
        # load lora adapters 
        # When resuming from a LoRA checkpoint the adapters default to inference mode,
        # make sure the PEFT wrapper re-enables gradients on those layers.
        try:
            model = PeftModel.from_pretrained(base_model, model_source, is_trainable=True)
        except TypeError:
            print("Peft could not set model to train, Manually doing it.")
            model = PeftModel.from_pretrained(base_model, model_source)
            peft_cfg = model.peft_config.get("default")
            if peft_cfg is not None:
                peft_cfg.inference_mode = False
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
    else:
        # full HF style mode from local dir i.e best_model dir, and prepares for PEFT 
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
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
    ##############################################################################################

    #sanity checks
    # Print trainable params
    model.print_trainable_parameters()

    # Any trainable params at all?
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_trainable > 0, "No trainable params — LoRA targets likely mismatched."

    # batch truly supervises tokens
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
    adapter_dir = sp.llama32_adv_adapter_model_dir if is_advanced else sp.llama32_adapter_model_dir
    best_model_dir = sp.llama32_adv_best_model_dir if is_advanced else sp.llama32_best_model_dir

    trainer.save_model(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"LoRA adapters saved to {adapter_dir}")

    print("Merging LoRA adapters into the base model (bf16) before saving...")
    merge_lora_adapter(bp.llama_model_name, adapter_dir, best_model_dir)
    print(f"[Merged] Saved adapters + processor to: {best_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Llama 3.2 vision model.")
    parser.add_argument(
        "--training-mode",
        choices=["basic", "advanced"],
        default="basic",
        help="Choose between the basic LoRA recipe or an advanced variant.",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        action="store_true",
        help="If set, initialize weights/processors from the latest best_model_dir/checkpoint instead of the HF base model.",
    )
    cli_args = parser.parse_args()
    main(training_mode=cli_args.training_mode, load_from_checkpoint=cli_args.load_from_checkpoint)
