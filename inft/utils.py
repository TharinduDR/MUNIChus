import os
from typing import Optional
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

def merge_lora_adapter(
    base_model_id: str,
    adapter_dir: str,
    output_dir: str,
    processor_source: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Merge a LoRA adapter checkpoint into its base model and persist a full HF model.
    """
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    processor_source = processor_source or adapter_dir
    target_dtype = torch.bfloat16
    resolved_device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    device_map = "auto" if resolved_device in {"auto", "cuda"} and torch.cuda.is_available() else None

    def _load_model(dtype: torch.dtype):
        return AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    try:
        base_model = _load_model(target_dtype)
        dtype_in_use = target_dtype
    except (ValueError, RuntimeError) as load_exc:
        # Environments without bfloat16 support (e.g., CPU only) need a fallback.
        print(f"Warning: unable to load base model in bf16 ({load_exc}). Falling back to float32.")
        base_model = _load_model(torch.float32)
        dtype_in_use = torch.float32

    if device_map is None:
        base_model = base_model.to(resolved_device)

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model = merged_model.to(dtype_in_use).to("cpu")

    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(processor_source, use_fast=True)
    processor.save_pretrained(output_dir)

    print(f"Merged model + processor saved to {output_dir}")

