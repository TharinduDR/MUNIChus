# inft Fine-Tuning Guide

This guide explains how the two supervised fine-tuning entry points in `inft/`—`llama32_inft.py` (Meta Llama 3.2 Vision Instruct) and `aya_inft.py` (Cohere Aya Vision 8B)—are structured, which configuration objects they depend on, and how to run or extend them for the MUNIChus multilingual captioning task.

## Shared pipeline
- **Dataset** – Both scripts call `dataloader.build_processed_dataset` with `MUNIChusLoadConfig`. The config defaults to the `tharindu/MUNIChus` dataset, trains on 10 languages (Arabic → Chinese), drops empty rows, and supports per-language caps for smoke tests. Batches are size 1 with multi-worker dataloading; adjust `batch_size`/`num_workers` inside `MUNIChusLoadConfig`.
- **Collation** – Vision/language batches are prepared by a model-specific collator (`LlamaCollator` or `AyaCollator`). They enforce `channels_last` tensors, respect `BasicParams.max_length` (4096), and omit generation prompts because supervision already supplies the desired response.
- **Processors & token padding** – `AutoProcessor.from_pretrained` loads the matching processor (`BasicParams.llama_processor_name` or `.aya_processor_name`). If the underlying tokenizer lacks a pad token, the scripts reuse `eos_token` to avoid `None` pads during collation.
- **Quantization (QLoRA)** – `QLoRAParams` drives a `BitsAndBytesConfig` (4-bit NF4 weights, optional double-quant, BF16 compute). Keep `torch.cuda.is_bf16_supported()` in mind; the scripts print CUDA capability before training.
- **LoRA adapters** – `LoRAParams` provides rank/alpha/dropout plus target modules. `training_mode=basic` restricts adapters to attention & MLP projections; `training_mode=advanced` widens coverage:  
  - **Llama** adds `mm_projector`, `multi_modal_projector`, `vision_proj`.  
  - **Aya** replaces those with projector-specific module names (`multi_modal_projector.linear_1/2`).  
  Modify these tuples if you inspect the model with `list_linear_modules` (helper in `aya_inft.py`) and discover additional linear layers worth adapting.
- **Model prep** – Both scripts:
  1. Load `AutoModelForImageTextToText` with `device_map="auto"` and `BasicParams.attn_impl` (defaults to `"eager"`).
  2. Run `prepare_model_for_kbit_training` to make the quantized weights trainable.
  3. Wrap with `get_peft_model` using the LoRA config.
  4. Disable `config.use_cache` so gradient checkpointing works.
- **Sanity guards** – Right after wrapping the model, the scripts print trainable-parameter counts, assert at least one parameter is trainable, and fetch a single collated batch to verify that some tokens carry supervision (`labels != -100`). Do not skip these checks; failures usually imply a broken collator or mismatched adapter target names.
- **Training loop** – `SFTTrainer` from TRL runs pure training (evaluation dataset is `None`). Core hyperparameters live in `SFTParams`: cosine LR schedule, 1.5e-4 LR, 16 gradient-accumulation steps (effective batch 16), weight decay 1e-6, gradient checkpointing, BF16, `max_grad_norm=0.3`, `save_steps=500`, and `save_total_limit=2`. Logs go to JSONL via `JsonlLogger` + `TrainerLoggingCallback`.
- **Outputs** – Checkpoints land in the mode-specific `SFTParams.*_output_dir`. After `trainer.train()`, the scripts copy the adapter weights (`trainer.model.save_pretrained`) and processor (`processor.save_pretrained`) into the `*_best_model_dir`. Point downstream inference at these directories to load PEFT adapters plus processor state.

## Running the scripts
Both entry points accept `--training-mode {basic,advanced}` (default basic). Everything else is controlled through the dataclasses in `inft/params.py`.

### Llama 3.2 Vision (`inft/llama32_inft.py`)
```bash
python inft/llama32_inft.py --training-mode basic
# or
python inft/llama32_inft.py --training-mode advanced
```
- Uses `BasicParams.llama_model_name`/`.llama_processor_name`.
- Collation via `LlamaCollator`.
- Default outputs:  
  - Basic → `SFTParams.llama_32_output_dir` / `llama32_best_model_dir`  
  - Advanced → `SFTParams.llama32_adv_output_dir` / `llama32_adv_best_model_dir`
- `SFTConfig` is constructed inline; modify learning rate, scheduler, or logging cadence inside `llama32_inft.py` if your hardware needs different settings.

### Aya Vision 8B (`inft/aya_inft.py`)
```bash
python inft/aya_inft.py --training-mode basic
# or
python inft/aya_inft.py --training-mode advanced
```
- Uses `BasicParams.aya_model_name`/`.aya_processor_name`.
- Collation via `AyaCollator`; a helper `list_linear_modules` exists to inspect PEFT targets (uncomment the call in advanced mode when you need a module inventory).
- Default outputs:  
  - Basic → `SFTParams.aya_output_dir` / `aya_best_model_dir`  
  - Advanced → `SFTParams.aya_adv_output_dir` / `aya_adv_best_model_dir`
- Shares the same SFT hyperparameters through `_build_sft_args`, so overrides should happen either inside `SFTParams` or by editing that helper.

## Logging & monitoring
- Hardware info (CUDA availability, GPU name, torch + CUDA build, BF16 support) prints at startup, making remote runs easier to triage.
- Trainer metrics—loss, learning rate, step timings—are appended to `{BasicParams.logs_dir}/trainer_{task_tag}.jsonl`. Stream the file or post-process it for dashboards.
- Because `report_to="none"`, no external tracking service is hit; integrate W&B or MLflow by adding the service name to `report_to` and ensuring credentials exist.

## Customization checklist
- **Dataset tweaks** – Edit `MUNIChusLoadConfig` for different splits, language subsets, or higher batch sizes (remember to adjust gradient accumulation accordingly).
- **Precision & attention** – Switch `BasicParams.attn_impl` (e.g., `"flash"`) if your GPU/driver supports it. Keep an eye on BF16 readiness; toggle to FP16 by changing `QLoRAParams.bnb_4bit_compute_dtype`.
- **Adapter coverage** – If advanced mode still leaves performance on the table, rerun `list_linear_modules` (Aya) or inspect `state_dict` keys (Llama) to expand the `advanced_target_modules`.
- **Validation** – Currently no eval split is wired in. To add validation, pass a processed eval dataset into `SFTTrainer` and set `evaluation_strategy`. Pair it with `load_best_model_at_end` (already enabled in `SFTParams`).
- **Safety checks** – Keep the “trainable params” and “supervised tokens” assertions enabled when modifying data or collators; they catch the most common configuration mistakes before wasting GPU hours.

With these components in place, you can consistently reproduce and extend the Aya and Llama supervised fine-tuning runs on the MUNIChus captioning corpus.
