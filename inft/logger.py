import os
import json
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from transformers import TrainerCallback, TrainingArguments


class _C:
    RESET="\x1b[0m"; DIM="\x1b[2m"; BOLD="\x1b[1m"
    GREEN="\x1b[32m"; YELLOW="\x1b[33m"; RED="\x1b[31m"; CYAN="\x1b[36m"; MAG="\x1b[35m"

def _utc_iso() -> str:
    return datetime.now().isoformat()

def _fmt_secs(s: float) -> str:
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


class JsonlLogger:
    """Minimal JSONL logger: writes one JSON object per line and mirrors to console."""
    def __init__(self, path: str, mirror_to_console: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.mirror = mirror_to_console
        # Small daily rotation by including date in filename (optional)
        base, ext = os.path.splitext(self.path)
        self._day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._active_path = f"{base}.{self._day}{ext or '.jsonl'}"

    def _rollover_if_needed(self):
        day_now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day_now != self._day:
            self._day = day_now
            base, ext = os.path.splitext(self.path)
            self._active_path = f"{base}.{self._day}{ext or '.jsonl'}"

    def write(self, record: Dict[str, Any]):
        self._rollover_if_needed()
        # Attach timestamp at write time
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False)
        with open(self._active_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.mirror:
            print(line)

class TrainerLoggingCallback(TrainerCallback):
    """Bridges Trainer events to our JsonlLogger."""
    def __init__(self, logger: JsonlLogger, run_name: str = "inft_run"):
        self.logger = logger
        self.run_name = run_name
        self._t0 = None

    # Called once at training start
    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()
        self.logger.write({
            "event": "train_begin",
            "run_name": self.run_name,
            "num_update_steps": state.max_steps,
            "world_size": args.world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "effective_batch_size": args.per_device_train_batch_size *
                                    args.gradient_accumulation_steps *
                                    max(1, args.world_size),
            "lr": args.learning_rate,
            "scheduler": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "bf16": args.bf16,
            "grad_clip": args.max_grad_norm,
        })

    # Fired every args.logging_steps with metrics
    def on_log(self, args, state, control, logs=None, **kwargs):
        # logs might include: loss, learning_rate, epoch, etc.
        rec = {
            "event": "log",
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "elapsed_sec": time.time() - self._t0 if self._t0 else None,
        }
        rec.update(logs or {})
        self.logger.write(rec)

    # Good time to note checkpoint writes
    def on_save(self, args, state, control, **kwargs):
        self.logger.write({
            "event": "checkpoint",
            "global_step": int(state.global_step),
            "save_dir": args.output_dir,
            "best_metric": state.best_metric,
        })

    # Called once at end
    def on_train_end(self, args, state, control, **kwargs):
        self.logger.write({
            "event": "train_end",
            "global_step": int(state.global_step),
            "epochs_trained": float(state.epoch) if state.epoch is not None else None,
            "total_flos": int(state.total_flos) if state.total_flos is not None else None,
            "elapsed_sec": time.time() - self._t0 if self._t0 else None,
        })
