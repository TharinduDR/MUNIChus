import json
import os
from dataclasses import replace, asdict
from typing import Dict, List, Optional, Tuple
from transformers.trainer_utils import set_seed
import torch
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from ayacollator import AyaCollator
from dataloader import build_processed_dataset
from llamacollator import LlamaCollator
from params import BasicParams, MUNIChusLoadConfig, SFTParams


def get_generation_kwargs(params: BasicParams) -> Dict:
    """Extract generation keyword arguments from BasicParams."""
    return {
        "do_sample": params.gen_do_sample,
        "top_p": params.gen_top_p,
        "temperature": params.gen_temperature,
        "num_beams": params.gen_num_beams,
        "max_new_tokens": params.gen_max_new_tokens,
    }

def _resolve_checkpoint_path(local_dir: str, hf_repo: str) -> str:
    """
    Prefer a locally fine-tuned checkpoint if it looks populated; fall back to the base repo otherwise.
    """
    if local_dir and os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir
    return hf_repo


def _load_model_and_processor( path: str, dtype: torch.dtype, base_model_name: str) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load either a full HF checkpoint or a LoRA adapter merged on top of the base model.
    """
    print(f"Loading model from path: {path} with dtype: {dtype}")
    adapter_config = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_config):
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, path)
    else:
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=dtype,
        )

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if getattr(model.config, "use_cache", None) is False:
        model.config.use_cache = True
    return model, processor


def load_llama32_model(prefer_sft: bool = True) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load the Llama 3.2 Vision model (fine-tuned checkpoint if available, otherwise HF base).
    Returns (model.eval(), processor).
    """
    basic = BasicParams()
    sft = SFTParams()

    checkpoint = _resolve_checkpoint_path(
        sft.llama32_best_model_dir if prefer_sft else "",
        basic.llama_model_name,
    )
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(getattr(basic, "dtype", "bfloat16"), torch.bfloat16)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(getattr(basic, "dtype", "bfloat16"), torch.bfloat16)
    return _load_model_and_processor(checkpoint, torch_dtype, basic.llama_model_name)


def load_aya_model(prefer_sft: bool = True) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load the Aya Vision 8B model (fine-tuned checkpoint if available, otherwise HF base).
    Returns (model.eval(), processor).
    """
    basic = BasicParams()
    sft = SFTParams()

    checkpoint = _resolve_checkpoint_path(
        sft.aya_best_model_dir if prefer_sft else "",
        basic.aya_model_name,
    )
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(getattr(basic, "dtype", "bfloat16"), torch.bfloat16)
    return _load_model_and_processor(checkpoint, torch_dtype, basic.aya_model_name)


def load_test_dataset(cfg:MUNIChusLoadConfig) -> Dataset:
    """
    Build the processed MUNIChus test split using the existing preprocessing pipeline.
    Returns (config_used, dataset).
    """
    cfg.split = "test"
    print(f"\nLoading test dataset with config: {asdict(cfg)}\n\n")
    if cfg.max_rows_per_lang is not None:
        cfg = replace(cfg, max_rows_per_lang=cfg.max_rows_per_lang)
    dataset = build_processed_dataset(cfg=cfg)
    return dataset



def llama32_gen_caption(cfg:MUNIChusLoadConfig, params:BasicParams):
    os.makedirs(params.logs_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)
    set_seed(getattr(params, "seed", 42))

    ds_test = load_test_dataset(cfg=cfg)
    print(f"Test dataset size: {len(ds_test)} examples")

    model, processor = load_llama32_model(prefer_sft=params.prefer_sft)

    collator = LlamaCollator(
        processor=processor,
        max_length=params.max_length,
        input_data_format="channels_last",
        add_generation_prompt=True,
    )
    dataloader = DataLoader(
        ds_test,
        batch_size=getattr(cfg, "batch_size", 1),
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        collate_fn=collator,
    )

    gen_kwargs = get_generation_kwargs(params)
    gen_kwargs.setdefault("eos_token_id", processor.tokenizer.eos_token_id)
    gen_kwargs.setdefault("pad_token_id", processor.tokenizer.pad_token_id)


    results: List[Dict[str, Optional[str]]] = []
    seen = 0
    try:
        for batch in dataloader:
            
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                "pixel_values": batch["pixel_values"].to(model.device, dtype=model.dtype),
                "aspect_ratio_ids": batch["aspect_ratio_ids"].to(model.device),
                "aspect_ratio_mask": batch["aspect_ratio_mask"].to(model.device),
            }
            if "pixel_attention_mask" in batch:
                inputs["pixel_attention_mask"] = batch["pixel_attention_mask"].to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)
            
            input_length = inputs["input_ids"].shape[1]
            texts = processor.tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            for i in range(len(texts)):
                example = ds_test[seen+i]
                record = {
                    "lang": example.get("language"),
                    "generated_caption": texts[i],
                    "reference_caption": example.get("target_text") or example.get("caption"),
                    "title": example.get("title"),
                }
                print(json.dumps(record, ensure_ascii=False))
                results.append(record)
            
            seen += len(texts)
    except Exception as exc:
        for example in ds_test:
            error_record = {
                "lang": example.get("language"),
                "generated_caption": None,
                "reference_caption": example.get("target_text") or example.get("caption"),
                "title": example.get("title"),
                "error": str(exc),
            }
            print(json.dumps(error_record, ensure_ascii=False))
            results.append(error_record)
    return results





def aya_gen_caption(cfg:MUNIChusLoadConfig, params:BasicParams):
    os.makedirs(params.logs_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)
    set_seed(getattr(params, "seed", 42))

    ds_test = load_test_dataset(cfg=cfg)
    print(f"Test dataset size: {len(ds_test)} examples")
    model, processor = load_aya_model(prefer_sft=params.prefer_sft)

    collator = AyaCollator(
        processor=processor,
        max_length=params.max_length,
        input_data_format="channels_last",
        add_generation_prompt=True,
    )

    dataloader = DataLoader(
        ds_test,
        batch_size=getattr(cfg, "batch_size", 1),
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        collate_fn=collator,
    )
    gen_kwargs = get_generation_kwargs(params)
    gen_kwargs.setdefault("eos_token_id", processor.tokenizer.eos_token_id)
    gen_kwargs.setdefault("pad_token_id", processor.tokenizer.pad_token_id)

    results: List[Dict[str, Optional[str]]] = []
    seen = 0
    try:
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                "pixel_values": batch["pixel_values"].to(model.device, dtype=model.dtype),
            }
            if "pixel_attention_mask" in batch:
                inputs["pixel_attention_mask"] = batch["pixel_attention_mask"].to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            input_length = inputs["input_ids"].shape[1]
            texts = processor.tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            for i in range(len(texts)):
                example = ds_test[seen+i]
                record = {
                    "lang": example.get("language"),
                    "generated_caption": texts[i],
                    "reference_caption": example.get("target_text") or example.get("caption"),
                    "title": example.get("title"),
                }
                print(json.dumps(record, ensure_ascii=False))
                results.append(record)

            seen += len(texts)
    except Exception as exc:
        for example in ds_test:
            error_record = {
                "lang": example.get("language"),
                "generated_caption": None,
                "reference_caption": example.get("target_text") or example.get("caption"),
                "title": example.get("title"),
                "error": str(exc),
            }
            print(json.dumps(error_record, ensure_ascii=False))
            results.append(error_record)
    return results



def save_generated_caption(file_full_path: str, records: List[Dict[str, Optional[str]]]) -> None:
    """Persist generation records to disk in JSONL format."""
    if not records:
        print(f"No records to save for path: {file_full_path}")
        return

    directory = os.path.dirname(file_full_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_full_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {file_full_path}")


def main() -> None:
    """Generate captions for both base HF models and persist the outputs."""
    print("======== CHECK Model is INFT local or HF, Max rows per lang ========")
    basic = BasicParams()
    cfg = MUNIChusLoadConfig(split="test", max_rows_per_lang=2)
    basic.prefer_sft = False

    output_root = os.path.join(basic.output_dir, "inft_gen") if basic.prefer_sft else os.path.join(basic.output_dir, "hf_gen")

    try:
        llama_records = llama32_gen_caption(cfg=cfg, params=basic) 
        
        save_generated_caption(
            os.path.join(output_root, "llama32_hf.jsonl"),
            llama_records,
        )
    except Exception as exc:
        print(f"Llama32 generation failed: {exc}")

    try:
        aya_records = aya_gen_caption(cfg=cfg, params=basic)
        save_generated_caption(
            os.path.join(output_root, "aya_hf.jsonl"),
            aya_records,
        )
    except Exception as exc:
        print(f"Aya generation failed: {exc}")


if __name__ == "__main__":
    main()
