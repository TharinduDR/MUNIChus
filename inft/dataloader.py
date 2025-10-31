from params import MUNIChusLoadConfig
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import List, Optional, Dict
from prompt_developer import to_supervised_record, build_instruction

def load_one_language(cfg: MUNIChusLoadConfig, lang:str)->Dataset:
    """
    Load a single-language subset. The dataset provides subsets keyed by language code.
    Columns: image (PIL), title (str), content (str), caption (str).
    """
    ds = load_dataset(cfg.dataset_id, lang, split=cfg.split, streaming=False)
    # consistent types and optionally filter 
    def _ok(example):
        if not cfg.remove_empty: return True
        return bool(example.get("image")) and bool(example.get("content")) and bool(example.get("caption"))
    
    ds = ds.filter(_ok)

    if cfg.max_rows_per_lang is not None and len(ds) > cfg.max_rows_per_lang:
        ds = ds.shuffle(seed=cfg.seed).select(range(cfg.max_rows_per_lang))
    
    # adding lang to route prompting 
    ds = ds.add_column("language", [lang]*len(ds))

    return ds



def load_munichus(cfg: MUNIChusLoadConfig)->Dataset:
    """
    Load and concatenate the selected language splits into a single Dataset with columns:
    image (PIL.Image), title (str), content (str), caption (str), language (str)
    """
    parts: List[Dataset] = []
    for lang in  cfg.languages:
        parts.append(load_one_language(cfg=cfg, lang=lang))
    
    if len(parts) == 1: return parts[0]
    return concatenate_datasets(parts)



def build_processed_dataset(cfg: MUNIChusLoadConfig)->Dataset:
    """
    Load, validate, and map raw rows to {image, input_text, target_text, language}.
    lang_display_map to control how {language} is rendered in the prompt,
    e.g., {'en': 'English', 'fr':'French', ...}. If None, the language code is used.
    """
    raw_ds = load_munichus(cfg)
    def _map_fn(example):
        display = cfg.language_names.get(example["language"])
        rec = to_supervised_record(example)
        rec["input_text"] = build_instruction(example, lang_display=display)
        return rec
    
    cols_keep = ["image", "input_text", "target_text", "language", "title"]
    ds = raw_ds.map(_map_fn, remove_columns=[c for c in raw_ds.column_names if c not in cols_keep])
    return ds
    


# if __name__== "__main__":
#     cfg = MUNIChusLoadConfig(
#         max_rows_per_lang=10,
#         split="train",
#     )
#     ds = build_processed_dataset(cfg)
#     print(f"Loaded dataset with {len(ds)} examples.")
#     print(ds[0])

