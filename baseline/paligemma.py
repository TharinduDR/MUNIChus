import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
import sys

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed for CJK tokenization")
    CJK_TOKENIZATION_AVAILABLE = False

# Load PaliGemma model
print("Loading PaliGemma model...")
model_id = "google/paligemma-3b-mix-224"  # or "google/paligemma-3b-pt-224" or "google/paligemma-3b-mix-448"
paligemma_processor = AutoProcessor.from_pretrained(model_id)
paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Initialize metrics
chrf_metric = CHRF()
cider_scorer = Cider()

# Language codes
languages = ["ar", "en", "fr", "hi", "id", "ja", "si", "ur", "yue", "zh"]

language_names = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "ja": "Japanese",
    "si": "Sinhala",
    "ur": "Urdu",
    "yue": "Cantonese",
    "zh": "Chinese"
}

# ISO 639-1 codes for PaliGemma prompts
iso_lang_codes = {
    "ar": "ar",  # Arabic
    "en": "en",  # English
    "fr": "fr",  # French
    "hi": "hi",  # Hindi
    "id": "id",  # Indonesian
    "ja": "ja",  # Japanese
    "si": "si",  # Sinhala
    "ur": "ur",  # Urdu
    "yue": "zh",  # Cantonese (use Chinese code)
    "zh": "zh"  # Chinese
}

# CJK languages that need special tokenization
CJK_LANGUAGES = ["ja", "zh", "yue"]


def tokenize_text(text, lang_code):
    """Tokenize text appropriately based on language"""
    if not CJK_TOKENIZATION_AVAILABLE or lang_code not in CJK_LANGUAGES:
        return text

    if lang_code in ["zh", "yue"]:
        try:
            return " ".join(jieba.cut(text))
        except:
            return " ".join(list(text))
    elif lang_code == "ja":
        try:
            mecab = MeCab.Tagger("-Owakati")
            return mecab.parse(text).strip()
        except:
            return " ".join(list(text))
    return text


def generate_caption_paligemma(image, lang_code):
    """Generate caption directly in target language using PaliGemma"""

    # Create language-specific prompt
    iso_code = iso_lang_codes[lang_code]
    prompt = f"caption {iso_code}"

    # Prepare inputs
    inputs = paligemma_processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="longest"
    ).to(paligemma_model.device)

    # Generate
    with torch.no_grad():
        generated_ids = paligemma_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    # Decode (skip the prompt tokens)
    generated_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
    caption = paligemma_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return caption


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate PaliGemma on a specific language with direct multilingual generation"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    # Load test set
    try:
        dataset = load_dataset(dataset_name, lang_code)
        test_data = dataset['test']

        if num_samples:
            test_data = test_data.select(range(min(num_samples, len(test_data))))

        print(f"Processing {len(test_data)} examples...")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None, None, None

    predictions = []
    references = []

    # Generate captions
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        try:
            # Generate caption directly in target language
            generated_caption = generate_caption_paligemma(
                example['image'],
                lang_code
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            # Print sample outputs
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append("")
            references.append([example['caption']])

    # Validate predictions
    valid_predictions = [p for p in predictions if p]
    if not valid_predictions:
        print(f"✗ No valid predictions for {lang_code}")
        return None, predictions, references

    print(f"\nGenerated {len(valid_predictions)}/{len(predictions)} valid captions")

    # Tokenize for CJK languages if needed
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"Tokenizing texts for {lang_code} (CJK language)...")
        tokenized_predictions = [tokenize_text(pred, lang_code) for pred in predictions]
        tokenized_references = [[tokenize_text(ref[0], lang_code)] for ref in references]
    else:
        tokenized_predictions = predictions
        tokenized_references = references

    try:
        # Calculate BLEU-4
        print(f"\nCalculating BLEU-4 for {lang_code}...")
        bleu_metric = BLEU(max_ngram_order=4)
        tokenized_references_transposed = [[ref[0] for ref in tokenized_references]]
        bleu_score = bleu_metric.corpus_score(tokenized_predictions, tokenized_references_transposed)

        # Calculate chrF
        print(f"Calculating chrF for {lang_code}...")
        references_transposed = [[ref[0] for ref in references]]
        chrf_score = chrf_metric.corpus_score(predictions, references_transposed)

        # Calculate CIDEr
        print(f"Calculating CIDEr for {lang_code}...")
        predictions_dict = {i: [pred] for i, pred in enumerate(tokenized_predictions)}
        references_dict = {i: refs for i, refs in enumerate(tokenized_references)}
        cider_score, _ = cider_scorer.compute_score(references_dict, predictions_dict)

        results = {
            "language": lang_code,
            "language_name": language_names[lang_code],
            "num_samples": len(predictions),
            "num_valid": len(valid_predictions),
            "bleu4": bleu_score.score,
            "chrf": chrf_score.score,
            "cider": cider_score * 100
        }

        print(f"\nResults for {language_names[lang_code]}:")
        print(f"  Valid captions: {len(valid_predictions)}/{len(predictions)}")
        print(f"  BLEU-4: {results['bleu4']:.2f}")
        print(f"  chrF:   {results['chrf']:.2f}")
        print(f"  CIDEr:  {results['cider']:.2f}")

        return results, predictions, references

    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return None, predictions, references


# Main evaluation loop
if __name__ == "__main__":
    # Configuration
    NUM_SAMPLES = None  # Set to a number for testing (e.g., 50), None for full dataset

    all_results = []
    all_predictions = {}
    all_references = {}

    print("Starting PaliGemma direct multilingual evaluation...")
    print(f"Model: {model_id}")
    print("=" * 80)

    for lang in languages:
        try:
            results, preds, refs = evaluate_language(
                lang,
                dataset_name="tharindu/MUNIChus",
                num_samples=NUM_SAMPLES
            )

            if results is not None:
                all_results.append(results)
                all_predictions[lang] = preds
                all_references[lang] = refs

        except Exception as e:
            print(f"✗ Error evaluating {lang}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_results:
        print("\n✗ No results generated!")
        sys.exit(1)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY - PaliGemma Direct Multilingual")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv("paligemma_evaluation_results.csv", index=False)
    print("\n✓ Results saved to paligemma_evaluation_results.csv")

    # Save detailed predictions
    with open("paligemma_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references,
            "config": {
                "model": model_id,
                "approach": "direct_multilingual"
            }
        }, f, ensure_ascii=False, indent=2)
    print("✓ Predictions saved to paligemma_predictions.json")

    # Calculate averages
    print("\nAverage Scores Across All Languages:")
    print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")

    # Print CJK-specific averages
    cjk_results = results_df[results_df['language'].isin(CJK_LANGUAGES)]
    if not cjk_results.empty:
        print("\nAverage Scores for CJK Languages (ja, zh, yue):")
        print(f"  Average BLEU-4: {cjk_results['bleu4'].mean():.2f}")
        print(f"  Average chrF:   {cjk_results['chrf'].mean():.2f}")
        print(f"  Average CIDEr:  {cjk_results['cider'].mean():.2f}")

    # Print non-CJK averages
    non_cjk_results = results_df[~results_df['language'].isin(CJK_LANGUAGES)]
    if not non_cjk_results.empty:
        print("\nAverage Scores for Non-CJK Languages:")
        print(f"  Average BLEU-4: {non_cjk_results['bleu4'].mean():.2f}")
        print(f"  Average chrF:   {non_cjk_results['chrf'].mean():.2f}")
        print(f"  Average CIDEr:  {non_cjk_results['cider'].mean():.2f}")