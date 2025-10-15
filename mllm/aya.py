import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU
from pycocoevalcap.cider.cider import Cider
from io import BytesIO
import requests

# Initialize pipeline
print("Loading Aya Vision 8B...")
pipe = pipeline(
    model="CohereLabs/aya-vision-8b",
    task="image-text-to-text",
    device_map="auto"
)

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


def pil_image_to_url(image):
    """Convert PIL Image to data URL for Aya Vision"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    # For Aya Vision, we can use a temporary file path or convert to base64 data URL
    import base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_base64}"


def generate_caption_aya(image, news_content, language):
    """Generate caption using Aya Vision 8B"""

    # Convert image to URL format
    image_url = pil_image_to_url(image)

    # Create prompt
    prompt = f"""Given this image and its news article, write a short caption for the image in {language_names[language]}. Try to identify people names, locations and organisations in the image linking it to the news article and include them in the image caption.

News Article:
{news_content[:500]}...

Write only the caption in {language_names[language]}, nothing else."""

    # Format message with the aya-vision chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        outputs = pipe(text=messages, max_new_tokens=100, return_full_text=False)
        caption = outputs[0]['generated_text'].strip()
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate Aya Vision model on a specific language"""

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
            generated_caption = generate_caption_aya(
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
print("FINAL RESULTS SUMMARY - AYA VISION 8B")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("aya_vision_evaluation_results.csv", index=False)
print("\n✓ Results saved to aya_vision_evaluation_results.csv")

# Save detailed predictions
with open("aya_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references
    }, f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to aya_vision_predictions.json")

# Calculate averages
print("\nAverage Scores:")
print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")