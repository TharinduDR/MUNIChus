import cohere
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
import base64
from io import BytesIO
from sacrebleu.metrics import BLEU
from pycocoevalcap.cider.cider import Cider
import time

# Initialize Cohere client
COHERE_API_KEY = "your-api-key-here"  # Replace with your API key
co = cohere.Client(COHERE_API_KEY)

# Initialize metrics
bleu_metric = BLEU(max_ngram_order=4)
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


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def generate_caption_cohere(image, news_content, language, max_retries=3):
    """Generate caption using Cohere Vision API"""

    # Convert image to base64
    image_base64 = image_to_base64(image)

    # Create prompt
    prompt = f"""Given this news article and image, write a short newspaper caption in {language_names[language]}.

News Article:
{news_content[:500]}...

Write only the caption in {language_names[language]}, nothing else."""

    # Call Cohere API with retries
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model="command-r-plus-08-2024",  # Cohere's vision model
                message=prompt,
                images=[image_base64]
            )

            caption = response.text.strip()
            return caption

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} due to error: {e}")
                time.sleep(2)  # Wait before retry
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return ""

    return ""


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate Cohere model on a specific language"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    # Load test set
    dataset = load_dataset(dataset_name, lang_code)
    test_data = dataset['test']

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

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

            # Print sample outputs
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

            # Rate limiting: sleep briefly between requests
            time.sleep(0.5)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])

    # Calculate BLEU-4
    print(f"\nCalculating BLEU-4 for {lang_code}...")
    references_transposed = [[ref[0] for ref in references]]
    bleu_score = bleu_metric.corpus_score(predictions, references_transposed)

    # Calculate CIDEr
    print(f"Calculating CIDEr for {lang_code}...")
    predictions_dict = {i: [pred] for i, pred in enumerate(predictions)}
    references_dict = {i: refs for i, refs in enumerate(references)}
    cider_score, _ = cider_scorer.compute_score(references_dict, predictions_dict)

    results = {
        "language": lang_code,
        "language_name": language_names[lang_code],
        "num_samples": len(predictions),
        "bleu4": bleu_score.score,
        "cider": cider_score * 100
    }

    print(f"\nResults for {language_names[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references


# Evaluate all languages
all_results = []
all_predictions = {}
all_references = {}

for lang in languages:
    try:
        results, preds, refs = evaluate_language(
            lang,
            dataset_name="tharindu/MUNIChus",
            num_samples=None  # Set to 10-100 for quick testing
        )
        all_results.append(results)
        all_predictions[lang] = preds
        all_references[lang] = refs

    except Exception as e:
        print(f"Error evaluating {lang}: {e}")
        continue

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Print summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY - COHERE")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("cohere_vision_evaluation_results.csv", index=False)
print("\n✓ Results saved to cohere_vision_evaluation_results.csv")

# Save detailed predictions
with open("cohere_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references
    }, f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to cohere_vision_predictions.json")

# Calculate averages
print("\nAverage Scores:")
print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")