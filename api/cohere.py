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
import sys

# Initialize Cohere client
COHERE_API_KEY = "your-api-key-here"
co = cohere.ClientV2(COHERE_API_KEY)

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


def generate_caption_cohere(image, news_content, language, max_retries=3):
    """Generate caption using Cohere command-a-vision-07-2025"""

    # Convert image to data URL
    image_url = image_to_data_url(image)
    if image_url is None:
        return ""

    # Create prompt
    prompt = f"""Given this image and its news article, write a short caption for the image in {language_names[language]}. Try to identify famous people, locations and organisations in the image linking it to the news article and include them in the image caption.


News Article:
{news_content[:500]}...

Write only the caption in {language_names[language]}, nothing else."""

    # Call Cohere API with retries
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
                                "image": image_url  # Correct format!
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
            return caption

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} due to error: {e}")
                time.sleep(2)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return ""

    return ""


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate Cohere model on a specific language"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    try:
        dataset = load_dataset(dataset_name, lang_code)
        test_data = dataset['test']

        if num_samples:
            test_data = test_data.select(range(min(num_samples, len(test_data))))

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

            time.sleep(0.5)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])

    valid_predictions = [p for p in predictions if p]
    if not valid_predictions:
        print(f"✗ No valid predictions for {lang_code}")
        return None, predictions, references

    try:
        # Calculate metrics
        references_transposed = [[ref[0] for ref in references]]
        bleu_score = bleu_metric.corpus_score(predictions, references_transposed)

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

    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")
        return None, predictions, references


# Evaluate all languages
all_results = []
all_predictions = {}
all_references = {}

for lang in languages:
    try:
        results, preds, refs = evaluate_language(
            lang,
            dataset_name="tharindu/MUNIChus",
            num_samples=10  # Start with 10 for testing
        )

        if results is not None:
            all_results.append(results)
            all_predictions[lang] = preds
            all_references[lang] = refs

    except Exception as e:
        print(f"✗ Error evaluating {lang}: {e}")
        continue

if not all_results:
    print("\n✗ No results generated!")
    sys.exit(1)

# Save results
results_df = pd.DataFrame(all_results)
print("\n" + "=" * 80)
print("FINAL RESULTS - COHERE command-a-vision-07-2025")
print("=" * 80)
print(results_df.to_string(index=False))

results_df.to_csv("cohere_vision_evaluation_results.csv", index=False)

with open("cohere_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({"predictions": all_predictions, "references": all_references},
              f, ensure_ascii=False, indent=2)

print(f"\nAverage BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"Average CIDEr:  {results_df['cider'].mean():.2f}")