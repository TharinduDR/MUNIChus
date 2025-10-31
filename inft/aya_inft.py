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


def _build_lora_config(lora: LoRAParams) -> LoraConfig:
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )


def _build_training_args(sp: SFTParams) -> TrainingArguments:
    return TrainingArguments(
        output_dir=sp.aya_output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        num_train_epochs=sp.num_epochs,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        torch_compile=False,
        logging_steps=sp.logging_steps,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=sp.save_total_limit,
        save_safetensors=True,
        report_to="none",
        eval_strategy="no",
        remove_unused_columns=False,
        seed=sp.seed,
        ddp_find_unused_parameters=False,
    )


def main() -> None:
    _log_device_info()

    base_params = BasicParams()
    qlora_params = QLoRAParams()
    lora_params = LoRAParams()
    sft_params = SFTParams()
    data_cfg = MUNIChusLoadConfig(split="train")

    set_seed(sft_params.seed)

    os.makedirs(base_params.logs_dir, exist_ok=True)
    os.makedirs(base_params.output_dir, exist_ok=True)

    task_tag = "qlora-lora-aya8b"
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
    model = prepare_model_for_kbit_training(model)

    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = _build_lora_config(lora_params)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = _build_training_args(sft_params)
    #sft_config = SFTConfig(packing=False, max_seq_length=base_params.max_length)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=base_params.max_length,
        callbacks=[trainer_cb],
    )

    print("Starting supervised fine-tuning...")
    train_result = trainer.train()
    print("Training complete. Metrics:", train_result.metrics)

    os.makedirs(sft_params.aya_best_model_dir, exist_ok=True)
    trainer.model.save_pretrained(sft_params.aya_best_model_dir)
    processor.save_pretrained(sft_params.aya_best_model_dir)
    print(f"Saved adapters + processor to: {sft_params.aya_best_model_dir}")


if __name__ == "__main__":
    main()
