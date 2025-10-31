"""
Per-language one-batch test for MUNIChus → Llama-3.2-Vision.
writes a SINGLE combined log file:
  <logs_dir>/llam32_pipeline_samples_combined.txt
"""

import os
from typing import Dict
from dataclasses import asdict, replace
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText

from params import MUNIChusLoadConfig, BasicParams
from dataloader import build_processed_dataset
from llamacollator import LlamaCollator


def sanity_check_train_batch(batch: Dict[str, torch.Tensor]):
    assert set(["input_ids","attention_mask","labels"]).issubset(batch.keys())
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    non_ignored = (batch["labels"] != -100).sum(dim=1)
    assert torch.all(non_ignored > 0), "TRAIN: found rows with zero supervised tokens."
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100), "Padding not masked."


def sanity_check_infer_batch(batch: Dict[str, torch.Tensor]):
    assert set(["input_ids","attention_mask","labels"]).issubset(batch.keys())
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert torch.all(batch["labels"] == -100), "INFER: labels must be all -100."


def _dtype_from_str(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def ensure_dirs(params: BasicParams):
    os.makedirs(params.logs_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)


def _one_lang_dataset(cfg: MUNIChusLoadConfig, lang: str):
    """
    Build a dataset for a single language, capped to exactly one batch if possible.
    """
    per_cfg = replace(
        cfg,
        languages=(lang,),
        max_rows_per_lang=cfg.batch_size if cfg.max_rows_per_lang is None
        else min(cfg.max_rows_per_lang, cfg.batch_size),
    )
    return build_processed_dataset(per_cfg)


def run_test(cfg: MUNIChusLoadConfig, params: BasicParams):
    ensure_dirs(params)

    # Single combined log file
    combined_log = os.path.join(params.logs_dir, "llama32_pipeline_samples_combined.txt")
    with open(combined_log, "w", encoding="utf-8") as flog:
        flog.write("# MUNIChus → Llama-3.2-Vision Pipeline Test Log\n")
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
    processor = AutoProcessor.from_pretrained(params.llama_processor_name, trust_remote_code=True)
    collator_train = LlamaCollator(
        processor=processor,
        max_length=params.max_length,
        input_data_format="channels_last",
        add_generation_prompt=False,
    )
    collator_infer = LlamaCollator(
        processor=processor,
        max_length=params.max_length,
        input_data_format="channels_last",
        add_generation_prompt=True,
    )

    # Optional model (load once)
    model = None
    if params.run_forward:
        print("Loading model once for all languages...")
        model = AutoModelForImageTextToText.from_pretrained(
            params.llama_model_name,
            torch_dtype=_dtype_from_str(params.dtype),
            device_map=params.device_map,
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

        dl_train = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collator_train
        )
        dl_infer = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collator_infer
        )

        # Batches + sanity
        train_batch = next(iter(dl_train))
        train_shapes = {k: tuple(v.shape) for k, v in train_batch.items() if isinstance(v, torch.Tensor)}
        print("TRAIN shapes:", train_shapes)
        sanity_check_train_batch(train_batch)

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
            assert train_loss is not None, "Expected non-empty loss for TRAIN batch."

            with torch.no_grad():
                out_infer = model(**{k: v.to(model.device) for k, v in infer_batch.items()})
            infer_loss = getattr(out_infer, "loss", None)
            print("Forward INFER loss:", infer_loss if infer_loss is not None else "<None>")
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
            for i in range(take):
                flog.write("---- SAMPLE ----\n")
                flog.write("LANG: " + ds[i]["language"] + "\n")
                flog.write("PROMPT:\n" + ds[i]["input_text"] + "\n\n")
                flog.write("TARGET:\n" + ds[i]["target_text"] + "\n\n")
            flog.write("-" * 78 + "\n")

        print(f"✓ Language '{lang}' logged.")
        passed += 1

    print("\n" + "=" * 72)
    print(f"Per-language one-batch tests complete: {passed}/{total_langs} languages passed.")
    print(f"Combined log written to: {combined_log}")


# Example direct run
if __name__ == "__main__":
    cfg = MUNIChusLoadConfig(
        # languages=("en","fr","hi"),  # keep default to test all
        split="test",
        max_rows_per_lang=2,   # overridden per-language to = batch_size
        batch_size=4,
        num_workers=2,
        remove_empty=True,
        seed=42,
    )
    params = BasicParams(
        run_forward=False,
        device_map=None,
        dtype="bfloat16",
        max_length=4096,
    )
    run_test(cfg, params)
