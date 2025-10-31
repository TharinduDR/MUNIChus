import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
from params import BasicParams, MUNIChusLoadConfig

if TYPE_CHECKING:
    from sacrebleu.metrics import BLEU, CHRF
    from pycocoevalcap.cider.cider import Cider

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

cfg = MUNIChusLoadConfig()

# copied form mllm/aya.py
def tokenize_text(text, lang_code):
    """Tokenize text appropriately based on language"""
    if not CJK_TOKENIZATION_AVAILABLE or lang_code not in cfg.cjk_tokens:
        # For non-CJK languages, return as is
        return text

    if lang_code in ["zh", "yue"]:
        # Chinese/Cantonese - use jieba
        try:
            return " ".join(jieba.cut(text))
        except Exception as e:
            print(f"Jieba tokenization failed: {e}, using character-level")
            return " ".join(list(text))

    elif lang_code == "ja":
        # Japanese - use MeCab
        try:
            mecab = MeCab.Tagger("-Owakati")  # Wakati mode (space-separated)
            tokenized = mecab.parse(text).strip()
            return tokenized
        except Exception as e:
            print(f"MeCab tokenization failed: {e}, using character-level")
            return " ".join(list(text))

    return text


def _load_generation_records(json_path: Path) -> List[Dict[str, Any]]:
    """Load generation records from a .json or .jsonl file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics input file not found: {json_path}")

    records: List[Dict[str, Any]] = []
    with json_path.open("r", encoding="utf-8") as fh:
        suffix = json_path.suffix.lower()
        if suffix == ".jsonl":
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL on line {line_no} in {json_path}") from exc
        else:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                fh.seek(0)
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON entry on line {line_no} in {json_path}") from exc
            else:
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = [data]
                else:
                    raise ValueError(f"Unsupported JSON content in {json_path}: expected object or list.")
    return records


def _compute_language_metrics(
    lang_code: str,
    predictions: List[str],
    references: List[str],
    tokenized_predictions: List[str],
    tokenized_references: List[str],
    bleu_metric: Any,
    chrf_metric: Any,
    cider_scorer: Any,
) -> Dict[str, Any]:
    """Compute BLEU, chrF, and CIDEr scores for a single language slice."""
    if not predictions:
        raise ValueError(f"No predictions available for language '{lang_code}'.")

    tokenized_refs_for_bleu = [tokenized_references]
    bleu_score = bleu_metric.corpus_score(tokenized_predictions, tokenized_refs_for_bleu)

    references_for_chrf = [references]
    chrf_score = chrf_metric.corpus_score(predictions, references_for_chrf)

    predictions_dict = {i: [pred] for i, pred in enumerate(tokenized_predictions)}
    references_dict = {i: [ref] for i, ref in enumerate(tokenized_references)}
    cider_score, _ = cider_scorer.compute_score(references_dict, predictions_dict)

    if not lang_code or lang_code == "overall":
        language_name = "All Languages"
    elif lang_code == "unknown":
        language_name = "Unknown"
    else:
        language_name = cfg.language_names.get(lang_code, lang_code)

    return {
        "language": lang_code or "overall",
        "language_name": language_name,
        "num_samples": len(predictions),
        "bleu4": round(bleu_score.score, 2),
        "chrf": round(chrf_score.score, 2),
        "cider": round(cider_score * 100, 2),
    }


def compute_metrics_from_json(json_path: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Compute BLEU-4, chrF, and CIDEr metrics from a generation results JSON/JSONL file.

    The file is expected to contain records with at least:
      - lang / language
      - generated_caption
      - reference_caption
    """
    path = Path(json_path)
    records = _load_generation_records(path)

    if not records:
        raise ValueError(f"No records read from {json_path}")

    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise ImportError(
            "sacrebleu is required to compute metrics. Install it with `pip install sacrebleu`."
        ) from exc

    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pycocoevalcap is required to compute CIDEr. Install it with "
            "`pip install git+https://github.com/salaniz/pycocoevalcap`."
        ) from exc

    bleu_metric = BLEU(max_ngram_order=4)
    chrf_metric = CHRF()
    cider_scorer = Cider()

    predictions: List[str] = []
    references: List[str] = []
    tokenized_predictions: List[str] = []
    tokenized_references: List[str] = []
    languages_seen: List[str] = []

    preds_by_lang: Dict[str, List[str]] = defaultdict(list)
    refs_by_lang: Dict[str, List[str]] = defaultdict(list)
    tok_preds_by_lang: Dict[str, List[str]] = defaultdict(list)
    tok_refs_by_lang: Dict[str, List[str]] = defaultdict(list)

    for record in records:
        lang_code = record.get("lang") or record.get("language") or ""
        pred_text = record.get("generated_caption") or record.get("prediction") or ""
        ref_text = record.get("reference_caption") or record.get("reference") or record.get("caption") or ""

        pred_text = pred_text.strip() if isinstance(pred_text, str) else ""
        ref_text = ref_text.strip() if isinstance(ref_text, str) else ""

        if not pred_text or not ref_text:
            continue

        tokenized_pred = tokenize_text(pred_text, lang_code)
        tokenized_ref = tokenize_text(ref_text, lang_code)

        predictions.append(pred_text)
        references.append(ref_text)
        tokenized_predictions.append(tokenized_pred)
        tokenized_references.append(tokenized_ref)
        languages_seen.append(lang_code)

        lang_key = lang_code or "unknown"
        preds_by_lang[lang_key].append(pred_text)
        refs_by_lang[lang_key].append(ref_text)
        tok_preds_by_lang[lang_key].append(tokenized_pred)
        tok_refs_by_lang[lang_key].append(tokenized_ref)

    if not predictions:
        raise ValueError(f"No valid prediction/reference pairs found in {json_path}")

    unique_languages = {lang or "unknown" for lang in languages_seen}
    overall_lang_code = "overall" if len(unique_languages) > 1 else next(iter(unique_languages))

    overall_metrics = _compute_language_metrics(
        lang_code=overall_lang_code,
        predictions=predictions,
        references=references,
        tokenized_predictions=tokenized_predictions,
        tokenized_references=tokenized_references,
        bleu_metric=bleu_metric,
        chrf_metric=chrf_metric,
        cider_scorer=cider_scorer,
    )

    if len(unique_languages) > 1:
        by_language: Dict[str, Dict[str, Any]] = {}
        for lang_code in unique_languages:
            lang_key = lang_code or "unknown"
            by_language[lang_key] = _compute_language_metrics(
                lang_code=lang_key,
                predictions=preds_by_lang[lang_key],
                references=refs_by_lang[lang_key],
                tokenized_predictions=tok_preds_by_lang[lang_key],
                tokenized_references=tok_refs_by_lang[lang_key],
                bleu_metric=bleu_metric,
                chrf_metric=chrf_metric,
                cider_scorer=cider_scorer,
            )
        overall_metrics["by_language"] = by_language

    return overall_metrics, predictions, references


def main() -> None:
    basic_params = BasicParams()

    generation_files = {
        "llama32_hf": Path(basic_params.llama_hf_gen_file),
        "aya_hf": Path(basic_params.aya_hf_gen_file),
        "llama32_inft": Path(basic_params.llama_inft_gen_file),
        "aya_inft": Path(basic_params.aya_inft_gen_file),
    }

    processed_any = False
    for tag, input_path in generation_files.items():
        if not input_path.exists():
            print(f"[metrics] Skipping {tag}: missing file at {input_path}")
            continue

        metrics, _, _ = compute_metrics_from_json(str(input_path))

        metrics_root = Path(basic_params.metric_dir)
        metrics_root.mkdir(parents=True, exist_ok=True)

        output_path = metrics_root / f"{tag}_metrics.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)


        print(f"[metrics] Saved {tag} metrics to {output_path}")
        processed_any = True

    if not processed_any:
        raise FileNotFoundError("No generation files found; please ensure caption generation has been run.")


if __name__ == "__main__":
    main()
