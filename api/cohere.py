import cohere
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
import base64
from io import BytesIO
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
import time
import sys

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

# Initialize Cohere client
COHERE_API_KEY = "your-api-key-here"
co = cohere.ClientV2(COHERE_API_KEY)

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

# CJK languages that need special tokenization
CJK_LANGUAGES = ["ja", "zh", "yue"]


def tokenize_text(text, lang_code):
    """Tokenize text appropriately based on language"""
    if not CJK_TOKENIZATION_AVAILABLE or lang_code not in CJK_LANGUAGES:
        return text

    if lang_code in ["zh", "yue"]:
        try:
            return " ".join(jieba.cut(text))
        except Exception as e:
            print(f"Jieba tokenization failed: {e}, using character-level")
            return " ".join(list(text))

    elif lang_code == "ja":
        try:
            mecab = MeCab.Tagger("-Owakati")
            tokenized = mecab.parse(text).strip()
            return tokenized
        except Exception as e:
            print(f"MeCab tokenization failed: {e}, using character-level")
            return " ".join(list(text))

    return text


def image_to_data_url(image):
    """Convert PIL Image to data URL"""
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"Error converting image: {e}")
        return None


def generate_caption_cohere(image, news_content, language, max_retries=5):
    """Generate caption using Cohere command-a-vision-07-2025"""

    # Convert image to data URL
    image_url = image_to_data_url(image)
    if image_url is None:
        return ""

    # Create improved prompt
    prompt = f"""You are writing a caption for a newspaper image.

Given the image and this news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {language_names[language]}.

Guidelines:
- Write in {language_names[language]} language only
- Keep it brief (10-12 words)
- Identify and include: people's names, locations, and organizations visible in the image
- Connect what you see in the image to the news context
- Use journalistic style (factual, clear, objective)
- Focus on the main subject of the image

Caption in {language_names[language]}:"""

    # Call Cohere API with retries and exponential backoff
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model="command-a-vision-07-2025",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_url
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            caption = response.message.content[0].text.strip()

            # Clean common prefixes
            prefixes = [
                f"Caption in {language_names[language]}:",
                "Caption:",
                "Here is the caption:",
                f"{language_names[language]} caption:",
                "The caption is:"
            ]

            for prefix in prefixes:
                if caption.lower().startswith(prefix.lower()):
                    caption = caption[len(prefix):].strip().lstrip(':').strip()

            return caption

        except Exception as e:
            error_str = str(e)

            # Check if it's a rate limit error (429)
            if "429" in error_str or "rate limit" in error_str.lower():
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} due to error: {error_str[:100]}")
                    time.sleep(2)
                else:
                    print(f"Failed after {max_retries} attempts: {error_str[:100]}")
                    return ""

    return ""


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate Cohere model on a specific language with BLEU-4, CIDEr, and chrF"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

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
            generated_caption = generate_caption_cohere(
                example['image'],
                example['content'],
                lang_code
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

            # Rate limiting - adjust based on your tier
            time.sleep(12)  # 12 seconds = 5 requests per minute for trial tier

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            time.sleep(12)

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
        # Calculate BLEU-4 with tokenized texts
        print(f"\nCalculating BLEU-4 for {lang_code}...")
        if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
            print(f"  Using proper tokenization for BLEU-4")

        bleu_metric = BLEU(max_ngram_order=4)
        tokenized_references_transposed = [[ref[0] for ref in tokenized_references]]
        bleu_score = bleu_metric.corpus_score(tokenized_predictions, tokenized_references_transposed)

        # Calculate chrF (doesn't need tokenization - works on characters)
        print(f"Calculating chrF for {lang_code}...")
        references_transposed = [[ref[0] for ref in references]]
        chrf_score = chrf_metric.corpus_score(predictions, references_transposed)

        # Calculate CIDEr with tokenized texts
        print(f"Calculating CIDEr for {lang_code}...")
        if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
            print(f"  Using proper tokenization for CIDEr")

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


# Evaluate all languages
all_results = []
all_predictions = {}
all_references = {}

print("Starting evaluation with rate limiting...")
print("Using 12 second delays between requests (5 requests/minute)")
print("=" * 80)

for lang in languages:
    try:
        results, preds, refs = evaluate_language(
            lang,
            dataset_name="tharindu/MUNIChus",
            num_samples=None  # Set to 50 for testing
        )

        if results is not None:
            all_results.append(results)
            all_predictions[lang] = preds
            all_references[lang] = refs

        # Extra wait between languages
        print(f"\nWaiting 30 seconds before next language...")
        time.sleep(30)

    except Exception as e:
        print(f"✗ Error evaluating {lang}: {e}")
        import traceback

        traceback.print_exc()
        continue

if not all_results:
    print("\n✗ No results generated!")
    sys.exit(1)

# Save results
results_df = pd.DataFrame(all_results)
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY - COHERE command-a-vision-07-2025")
print("=" * 80)
print(results_df.to_string(index=False))

results_df.to_csv("cohere_vision_evaluation_results.csv", index=False)
print("\n✓ Results saved to cohere_vision_evaluation_results.csv")

with open("cohere_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({"predictions": all_predictions, "references": all_references},
              f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to cohere_vision_predictions.json")

# Calculate averages
print("\nAverage Scores Across All Languages:")
print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")

# Print CJK-specific averages
cjk_results = results_df[results_df['language'].isin(CJK_LANGUAGES)]
if not cjk_results.empty:
    print("\nAverage Scores for CJK Languages (ja, zh, yue) - with proper tokenization:")
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