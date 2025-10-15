import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed for CJK tokenization")
    CJK_TOKENIZATION_AVAILABLE = False

# Load BLIP model (English captioning)
print("Loading BLIP model for English captioning...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load NLLB translation model
print("Loading NLLB-200 translation model...")
nllb_model_name = "facebook/nllb-200-3.3B"  # or "facebook/nllb-200-3.3B" for better quality
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to("cuda" if torch.cuda.is_available() else "cpu")

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

# NLLB language codes mapping
nllb_lang_codes = {
    "ar": "arb_Arab",  # Arabic
    "en": "eng_Latn",  # English
    "fr": "fra_Latn",  # French
    "hi": "hin_Deva",  # Hindi
    "id": "ind_Latn",  # Indonesian
    "ja": "jpn_Jpan",  # Japanese
    "si": "sin_Sinh",  # Sinhala
    "ur": "urd_Arab",  # Urdu
    "yue": "yue_Hant",  # Cantonese (Traditional)
    "zh": "zho_Hans"  # Chinese (Simplified)
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


def generate_english_caption(image):
    """Generate English caption using BLIP"""
    inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)

    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)

    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


def translate_caption(text, target_lang_code):
    """Translate English caption to target language using NLLB"""
    if target_lang_code == "en":
        return text  # No translation needed

    # Set source and target languages
    nllb_tokenizer.src_lang = nllb_lang_codes["en"]
    target_nllb_code = nllb_lang_codes[target_lang_code]

    # Tokenize
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True).to(nllb_model.device)

    # Translate
    with torch.no_grad():
        translated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.lang_code_to_id[target_nllb_code],
            max_new_tokens=100
        )

    # Decode
    translated_text = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text


def generate_caption_with_translation(image, target_lang_code):
    """Generate English caption and translate to target language"""
    # Step 1: Generate English caption
    english_caption = generate_english_caption(image)

    # Step 2: Translate to target language
    if target_lang_code == "en":
        return english_caption

    translated_caption = translate_caption(english_caption, target_lang_code)
    return translated_caption


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate BLIP+NLLB pipeline on a specific language"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    # Load test set
    dataset = load_dataset(dataset_name, lang_code)
    test_data = dataset['test']

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    print(f"Processing {len(test_data)} examples...")

    predictions = []
    english_captions = []  # Store English captions for analysis
    references = []

    # Generate captions
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        try:
            # Generate English caption
            eng_caption = generate_english_caption(example['image'])
            english_captions.append(eng_caption)

            # Translate to target language
            translated_caption = translate_caption(eng_caption, lang_code)

            predictions.append(translated_caption)
            references.append([example['caption']])

            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"English:    {eng_caption}")
                print(f"Translated: {translated_caption}")
                print(f"Reference:  {example['caption']}")
                print("-" * 80)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            english_captions.append("")
            references.append([example['caption']])

    # Tokenize for CJK languages if needed
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"Tokenizing texts for {lang_code} (CJK language)...")
        tokenized_predictions = [tokenize_text(pred, lang_code) for pred in predictions]
        tokenized_references = [[tokenize_text(ref[0], lang_code)] for ref in references]
    else:
        tokenized_predictions = predictions
        tokenized_references = references

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
        "bleu4": bleu_score.score,
        "chrf": chrf_score.score,
        "cider": cider_score * 100
    }

    print(f"\nResults for {language_names[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references, english_captions


# Evaluate all languages
all_results = []
all_predictions = {}
all_references = {}
all_english_captions = {}

for lang in languages:
    try:
        results, preds, refs, eng_caps = evaluate_language(
            lang,
            dataset_name="tharindu/MUNIChus",
            num_samples=None  # Set to 10-100 for quick testing
        )
        all_results.append(results)
        all_predictions[lang] = preds
        all_references[lang] = refs
        all_english_captions[lang] = eng_caps

    except Exception as e:
        print(f"Error evaluating {lang}: {e}")
        import traceback

        traceback.print_exc()
        continue

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Print summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY - BLIP + NLLB Translation")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("blip_nllb_evaluation_results.csv", index=False)
print("\n✓ Results saved to blip_nllb_evaluation_results.csv")

# Save detailed predictions including English captions
with open("blip_nllb_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references,
        "english_captions": all_english_captions
    }, f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to blip_nllb_predictions.json")

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