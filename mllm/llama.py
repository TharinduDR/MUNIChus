import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from evaluate import load
import json

# Load model
print("Loading Llama-3.2-11B-Vision-Instruct...")
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load metrics
bleu = load("bleu")
cider = load("cider")

# Language codes
languages = ["ar", "en", "fr", "hi", "id", "ja", "si", "ur", "yue", "zh"]

# Language names for prompts
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


def generate_caption(image, news_content, language):
    """Generate caption using Llama-3.2-11B-Vision"""

    # Create prompt with news content and language specification
    prompt = f"""Given this news article and image, write a short newspaper caption in {language_names[language]}.

News Article:
{news_content[:500]}...

Write only the caption in {language_names[language]}, nothing else:"""

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    generated_text = processor.decode(output[0], skip_special_tokens=True)

    # Extract only the caption (after the prompt)
    # The model returns the full conversation, so we need to extract just the response
    if "assistant" in generated_text.lower():
        caption = generated_text.split("assistant")[-1].strip()
    else:
        caption = generated_text.split(prompt)[-1].strip()

    return caption


def evaluate_language(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None):
    """Evaluate model on a specific language"""

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
            # Generate caption
            generated_caption = generate_caption(
                example['image'],
                example['content'],
                lang_code
            )

            predictions.append(generated_caption)
            references.append([example['caption']])  # BLEU/CIDEr expect list of references

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

    # Calculate metrics
    print(f"\nCalculating metrics for {lang_code}...")

    # BLEU-4
    bleu_score = bleu.compute(predictions=predictions, references=references, max_order=4)

    # CIDEr
    # CIDEr expects dict format
    predictions_dict = {i: [pred] for i, pred in enumerate(predictions)}
    references_dict = {i: refs for i, refs in enumerate(references)}
    cider_score = cider.compute(predictions=predictions_dict, references=references_dict)

    results = {
        "language": lang_code,
        "language_name": language_names[lang_code],
        "num_samples": len(predictions),
        "bleu4": bleu_score['bleu'] * 100,  # Convert to percentage
        "cider": cider_score['score']
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
            num_samples=None  # Set to a number like 100 for quick testing
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
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("llama_vision_evaluation_results.csv", index=False)
print("\n✓ Results saved to llama_vision_evaluation_results.csv")

# Save detailed predictions
with open("llama_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references
    }, f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to llama_vision_predictions.json")

# Calculate average scores
print("\nAverage Scores:")
print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")