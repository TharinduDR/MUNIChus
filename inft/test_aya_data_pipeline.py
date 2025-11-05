"""
Per-language one-batch test for MUNIChus → Aya Vision (CohereLabs/aya-vision-8b).

What it checks for EACH language:
1) Data load via your pipeline (image, input_text, target_text).
2) Collator correctness:
   - TRAIN (add_generation_prompt=False): mask prompt+padding with -100, supervise caption.
   - INFER (add_generation_prompt=True): labels all -100.
3) Optional forward pass with AutoModelForImageTextToText.
4) Writes ONE combined log file with per-language sections.
"""

import os
from dataclasses import asdict, replace
from datetime import datetime
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText


from params import MUNIChusLoadConfig, BasicParams
from dataloader import build_processed_dataset
from ayacollator import AyaCollator


# -------------------------
# Sanity checks 
# -------------------------
def sanity_check_train_batch(batch: Dict[str, torch.Tensor]):
    req = {"input_ids", "attention_mask", "labels"}
    missing = req - set(batch.keys())
    assert not missing, f"Missing keys in TRAIN batch: {missing}"
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    
    non_ignored = (batch["labels"] != -100).sum(dim=1)
    zeros = (non_ignored == 0).sum().item()
    if zeros > 0:
        total = batch["labels"].size(0)
        rate = 100.0 * zeros / total
        print(f"[warn] TRAIN: {zeros}/{total} items ({rate:.1f}%) have ZERO supervised tokens.")
    # Padding masked
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100), "Padding not masked (-100)."


def sanity_check_infer_batch(batch: Dict[str, torch.Tensor]):
    req = {"input_ids", "attention_mask", "labels"}
    missing = req - set(batch.keys())
    assert not missing, f"Missing keys in INFER batch: {missing}"
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert torch.all(batch["labels"] == -100), "INFER labels must be all -100."


def _dtype_from_str(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def ensure_dirs(params: BasicParams):
    os.makedirs(params.logs_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)


def _one_lang_dataset(cfg: MUNIChusLoadConfig, lang: str):
    """Clone cfg to a single language and cap to exactly one batch if possible."""
    per_cfg = replace(
        cfg,
        languages=(lang,),
        max_rows_per_lang=cfg.batch_size if cfg.max_rows_per_lang is None
        else min(cfg.max_rows_per_lang, cfg.batch_size),
    )
    return build_processed_dataset(per_cfg)


# -------------------------
#  Test
# -------------------------
def run_test(cfg: MUNIChusLoadConfig, params: BasicParams):
    ensure_dirs(params)

    # Model/processor IDs (allow missing Aya-specific fields by falling back)
    aya_model = getattr(params, "aya_model_name", "CohereLabs/aya-vision-8b")
    aya_proc  = getattr(params, "aya_processor_name", aya_model)

    # One combined log file
    combined_log = os.path.join(params.logs_dir, "aya_pipeline_samples_combined.txt")
    with open(combined_log, "w", encoding="utf-8") as flog:
        flog.write("# MUNIChus → Aya Vision Pipeline Test Log\n")
        flog.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
        flog.write("## Global Config\n")
        flog.write(str(asdict(cfg)) + "\n")
        flog.write("## Params (subset)\n")
        flog.write(str({k: v for k, v in asdict(params).items() if k not in ["project_root_dir"]}) + "\n")
        flog.write("\n" + "=" * 78 + "\n")

    print("=== GLOBAL CFG ===")
    print(asdict(cfg))
    print("=== PARAMS ===")
    print({k: v for k, v in asdict(params).items() if k not in ["project_root_dir"]})
    print("=" * 72)
    print(f"Logging to: {combined_log}")

    # Processor + Collators reused across languages
    processor = AutoProcessor.from_pretrained(aya_proc, trust_remote_code=True)
    collator_train = AyaCollator(
        processor=processor,
        max_length=getattr(params, "max_length", 4096),
        input_data_format="channels_last",
        add_generation_prompt=False,
    )
    collator_infer = AyaCollator(
        processor=processor,
        max_length=getattr(params, "max_length", 4096),
        input_data_format="channels_last",
        add_generation_prompt=True,
    )

    # Optional model (load once)
    model = None
    if getattr(params, "run_forward", False):
        print("Loading Aya model once for all languages...")
        model = AutoModelForImageTextToText.from_pretrained(
            aya_model,
            torch_dtype=_dtype_from_str(getattr(params, "dtype", "bfloat16")),
            device_map=getattr(params, "device_map", None),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        print("Model loaded.")

    total_langs = len(cfg.languages)
    passed = 0

    for lang in cfg.languages:
        print(f"\n>>> Language: {lang}")
        print("-" * 72)

        ds = _one_lang_dataset(cfg, lang)
        if len(ds) == 0:
            print(f"[SKIP] No samples after filtering for language '{lang}'.")
            with open(combined_log, "a", encoding="utf-8") as flog:
                flog.write(f"\n### Language: {lang}\n")
                flog.write("[SKIP] No samples after filtering.\n")
                flog.write("-" * 78 + "\n")
            continue

        print(f"Loaded {len(ds)} samples for '{lang}'. Batch size target: {cfg.batch_size}")
        print("Prompt preview:\n", ds[0]["input_text"][:220])
        print("Target preview:\n", ds[0]["target_text"][:160])

        # DataLoaders (one batch per mode)
        dl_train = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collator_train
        )
        dl_infer = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collator_infer
        )

        # TRAIN sanity
        train_batch = next(iter(dl_train))
        train_shapes = {k: tuple(v.shape) for k, v in train_batch.items() if isinstance(v, torch.Tensor)}
        print("TRAIN shapes:", train_shapes)
        sanity_check_train_batch(train_batch)

        # INFER sanity
        infer_batch = next(iter(dl_infer))
        infer_shapes = {k: tuple(v.shape) for k, v in infer_batch.items() if isinstance(v, torch.Tensor)}
        print("INFER shapes:", infer_shapes)
        sanity_check_infer_batch(infer_batch)

        # Optional forward
        train_loss, infer_loss = None, None
        if model is not None:
            with torch.no_grad():
                out_train = model(**{k: v.to(model.device) for k, v in train_batch.items()})
            train_loss = getattr(out_train, "loss", None)
            print("Forward TRAIN loss:", train_loss if train_loss is not None else "<None>")
            # In training mode with labels, we expect a defined loss (may be very small if many zeros)
            assert train_loss is not None, "Expected non-empty loss for TRAIN batch."

            with torch.no_grad():
                out_infer = model(**{k: v.to(model.device) for k, v in infer_batch.items()})
            infer_loss = getattr(out_infer, "loss", None)
            print("Forward INFER loss:", infer_loss if infer_loss is not None else "<None>")
            # In inference path, labels are all -100; allow None or ~0 loss
            if infer_loss is not None:
                assert float(infer_loss) == 0.0 or float(infer_loss) < 1e-6

        # Append to combined log
        take = min(cfg.batch_size, len(ds))
        with open(combined_log, "a", encoding="utf-8") as flog:
            flog.write(f"\n### Language: {lang}\n")
            flog.write(f"Samples: {len(ds)} | BatchSize: {cfg.batch_size}\n")
            flog.write(f"TRAIN shapes: {train_shapes}\n")
            flog.write(f"INFER shapes: {infer_shapes}\n")
            if train_loss is not None:
                flog.write(f"Forward TRAIN loss: {train_loss}\n")
            if infer_loss is not None:
                flog.write(f"Forward INFER loss: {infer_loss}\n")
            flog.write("\n-- Samples --\n")
            for i in range(take):
                flog.write("---- SAMPLE ----\n")
                flog.write("LANG: " + ds[i]["language"] + "\n")
                flog.write("PROMPT:\n" + ds[i]["input_text"] + "\n\n")
                flog.write("TARGET:\n" + ds[i]["target_text"] + "\n\n")
            flog.write("-" * 78 + "\n")

        print(f"✓ Language '{lang}' logged.")
        passed += 1

    print("\n" + "=" * 72)
    print(f"Per-language one-batch Aya tests complete: {passed}/{total_langs} languages passed.")
    print(f"Combined log written to: {combined_log}")


# Example direct run
if __name__ == "__main__":
    cfg = MUNIChusLoadConfig(
        split="test",
        max_rows_per_lang=2,   
        batch_size=4,
        num_workers=2,
        remove_empty=True,
        seed=42,
    )
    
    params = BasicParams(
        device_map=None,
        dtype="bfloat16",
        max_length=4096,
    )
    run_test(cfg, params)
