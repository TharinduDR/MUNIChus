import os
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
import base64
from io import BytesIO
import time

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

# Initialize OpenAI client
print("Initializing OpenAI client...")
OPENAI_API_KEY = "your-api-key-here"

client = OpenAI(api_key=OPENAI_API_KEY)

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


def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_caption_gpt4(image, news_content, language, max_retries=3):
    """Generate caption using GPT-4 Vision"""

    # Encode image to base64
    image_base64 = encode_image_to_base64(image)

    # Create prompt
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

    # Call OpenAI API with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=100
            )

            # Extract response
            caption = response.choices[0].message.content.strip()

            # Clean common prefixes
            prefixes = [
                f"Caption in {language_names[language]}:",
                "Caption:",
                "Here is the caption:",
                f"{language_names[language]} caption:"
            ]

            for prefix in prefixes:
                if caption.lower().startswith(prefix.lower()):
                    caption = caption[len(prefix):].strip().lstrip(':').strip()

            return caption

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4 seconds
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return ""

    return ""


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate GPT-4 Vision model on a specific language with BLEU-4, CIDEr, and chrF"""

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
    references = []

    # Generate captions
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        try:
            generated_caption = generate_caption_gpt4(
                example['image'],
                example['content'],
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

            # Rate limiting - adjust based on your tier
            time.sleep(1)  # 1 second between requests

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            time.sleep(1)

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
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"  Using proper tokenization for BLEU-4")

    bleu_metric = BLEU(max_ngram_order=4)
    tokenized_references_transposed = [[ref[0] for ref in tokenized_references]]
    bleu_score = bleu_metric.corpus_score(tokenized_predictions, tokenized_references_transposed)

    # Calculate chrF
    print(f"Calculating chrF for {lang_code}...")
    references_transposed = [[ref[0] for ref in references]]
    chrf_score = chrf_metric.corpus_score(predictions, references_transposed)

    # Calculate CIDEr
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
        "bleu4": bleu_score.score,
        "chrf": chrf_score.score,
        "cider": cider_score * 100
    }

    print(f"\nResults for {language_names[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references


# Evaluate all languages
all_results = []
all_predictions = {}
all_references = {}

print("Starting evaluation with GPT-4 Vision...")
print("Note: This will incur API costs. Monitor your usage at platform.openai.com")
print("=" * 80)

for lang in languages:
    try:
        results, preds, refs = evaluate_language(
            lang,
            dataset_name="tharindu/MUNIChus",
            num_samples=None  # Set to 50 to control costs
        )
        all_results.append(results)
        all_predictions[lang] = preds
        all_references[lang] = refs

    except Exception as e:
        print(f"Error evaluating {lang}: {e}")
        import traceback

        traceback.print_exc()
        continue

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Print summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY - GPT-4 VISION")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("gpt4_vision_evaluation_results.csv", index=False)
print("\n✓ Results saved to gpt4_vision_evaluation_results.csv")

# Save detailed predictions
with open("gpt4_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references
    }, f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to gpt4_vision_predictions.json")

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

# Estimate costs
total_requests = sum([len(preds) for preds in all_predictions.values()])
print(f"\n💰 Cost Estimate:")
print(f"  Total requests: {total_requests}")
print(f"  Estimated cost (gpt-4o): ${total_requests * 0.005:.2f} - ${total_requests * 0.015:.2f}")
print(f"  Note: Actual costs may vary. Check platform.openai.com for exact billing")