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
import sys
import random
import numpy as np

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

# Global cache for training datasets
TRAIN_DATASETS_CACHE = {}


def load_train_dataset(lang_code, dataset_name="tharindu/MUNIChus"):
    """Load and cache training dataset for a language"""
    if lang_code not in TRAIN_DATASETS_CACHE:
        try:
            print(f"Loading training dataset for {language_names[lang_code]}...")
            dataset = load_dataset(dataset_name, lang_code)
            TRAIN_DATASETS_CACHE[lang_code] = dataset['train']
            print(f"  Loaded {len(TRAIN_DATASETS_CACHE[lang_code])} training examples for {lang_code}")
        except Exception as e:
            print(f"Failed to load training dataset for {lang_code}: {e}")
            TRAIN_DATASETS_CACHE[lang_code] = None

    return TRAIN_DATASETS_CACHE[lang_code]


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


def get_random_few_shot_examples(lang_code, num_examples=3, exclude_index=None):
    """Get random few-shot examples from the training set"""
    train_data = load_train_dataset(lang_code)

    if train_data is None or len(train_data) < num_examples:
        print(f"Warning: Not enough training data for {lang_code}")
        return []

    # Get random indices, excluding the current test example if needed
    available_indices = list(range(len(train_data)))
    if exclude_index is not None and exclude_index < len(train_data):
        available_indices.remove(exclude_index)

    # Sample random indices
    selected_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    examples = []
    for idx in selected_indices:
        try:
            example = train_data[idx]
            examples.append({
                'image': example['image'],
                'content': example['content'][:800],  # Truncate to keep prompt manageable
                'caption': example['caption']
            })
        except Exception as e:
            print(f"Error loading example {idx}: {e}")
            continue

    return examples


def generate_caption_gpt4_few_shot(image, news_content, language, few_shot_examples, max_retries=3):
    """Generate caption using GPT-4 Vision with few-shot examples"""

    # Encode target image to base64
    target_image_base64 = encode_image_to_base64(image)

    # Build message content with few-shot examples
    message_content = []

    # Add introduction
    message_content.append({
        "type": "text",
        "text": f"You are writing captions for newspaper images in {language_names[language]}.\n\nHere are some examples of good image captions:\n"
    })

    # Add few-shot examples
    for i, example in enumerate(few_shot_examples, 1):
        example_image_base64 = encode_image_to_base64(example['image'])

        message_content.append({
            "type": "text",
            "text": f"\nExample {i}:"
        })
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{example_image_base64}"
            }
        })
        message_content.append({
            "type": "text",
            "text": f"News excerpt: {example['content']}\nCaption: {example['caption']}\n"
        })

    # Add the target task
    message_content.append({
        "type": "text",
        "text": f"\nNow, write a caption for this new image:"
    })
    message_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{target_image_base64}"
        }
    })

    # Add final prompt
    final_prompt = f"""Given the news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {language_names[language]}.

Guidelines:
- Write in {language_names[language]} language only
- Keep it brief (10-12 words)
- Follow the style of the examples provided
- Identify and include: people's names, locations, and organizations visible
- Connect what you see to the news context
- Use journalistic style (factual, clear, objective)

Caption in {language_names[language]}:"""

    message_content.append({
        "type": "text",
        "text": final_prompt
    })

    # Call OpenAI API with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": message_content
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
                f"{language_names[language]} caption:",
                "The caption is:"
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


def evaluate_language_few_shot(lang_code, dataset_name="tharindu/MUNIChus", num_samples=None, num_few_shot=3):
    """Evaluate GPT-4 Vision model on a specific language with few-shot learning"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code}) with {num_few_shot}-shot learning")
    print(f"{'=' * 80}")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load test dataset
    try:
        dataset = load_dataset(dataset_name, lang_code)
        test_data = dataset['test']

        if num_samples:
            test_data = test_data.select(range(min(num_samples, len(test_data))))

        print(f"Processing {len(test_data)} test examples...")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None, None, None

    # Pre-load training dataset for this language
    train_data = load_train_dataset(lang_code)
    if train_data is None:
        print(f"✗ No training data available for {lang_code}, skipping...")
        return None, None, None

    predictions = []
    references = []

    # Generate captions with few-shot examples
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code} (few-shot)")):
        try:
            # Get random few-shot examples from training set
            few_shot_examples = get_random_few_shot_examples(
                lang_code,
                num_examples=num_few_shot,
                exclude_index=None  # Could exclude if test/train overlap is concern
            )

            if len(few_shot_examples) < num_few_shot:
                print(f"Warning: Only {len(few_shot_examples)} examples available for few-shot")

            # Generate caption with few-shot examples
            generated_caption = generate_caption_gpt4_few_shot(
                example['image'],
                example['content'],
                lang_code,
                few_shot_examples
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            # Print first few examples
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Using {len(few_shot_examples)} few-shot examples")
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
            "num_few_shot": num_few_shot,
            "bleu4": bleu_score.score,
            "chrf": chrf_score.score,
            "cider": cider_score * 100
        }

        print(f"\nResults for {language_names[lang_code]} ({num_few_shot}-shot):")
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
    NUM_FEW_SHOT = 3  # Number of few-shot examples to use
    NUM_SAMPLES = None  # Set to a number for testing (e.g., 50), None for full dataset

    all_results = []
    all_predictions = {}
    all_references = {}

    print(f"Starting {NUM_FEW_SHOT}-shot evaluation with GPT-4 Vision...")
    print("Note: This will incur API costs. Monitor your usage at platform.openai.com")
    print("=" * 80)

    for lang in languages:
        try:
            results, preds, refs = evaluate_language_few_shot(
                lang,
                dataset_name="tharindu/MUNIChus",
                num_samples=NUM_SAMPLES,
                num_few_shot=NUM_FEW_SHOT
            )

            if results is not None:
                all_results.append(results)
                all_predictions[lang] = preds
                all_references[lang] = refs

            # Extra wait between languages
            print(f"\nWaiting 5 seconds before next language...")
            time.sleep(5)

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
    print(f"FINAL RESULTS SUMMARY - GPT-4 VISION ({NUM_FEW_SHOT}-shot learning)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    output_prefix = f"gpt4_vision_{NUM_FEW_SHOT}shot"

    results_df.to_csv(f"{output_prefix}_evaluation_results.csv", index=False)
    print(f"\n✓ Results saved to {output_prefix}_evaluation_results.csv")

    with open(f"{output_prefix}_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references,
            "config": {
                "num_few_shot": NUM_FEW_SHOT,
                "model": "gpt-4o"
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Predictions saved to {output_prefix}_predictions.json")

    # Calculate averages
    print("\nAverage Scores Across All Languages:")
    print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")

    # Print CJK-specific averages
    cjk_results = results_df[results_df['language'].isin(CJK_LANGUAGES)]
    if not cjk_results.empty:
        print(f"\nAverage Scores for CJK Languages (ja, zh, yue) - {NUM_FEW_SHOT}-shot:")
        print(f"  Average BLEU-4: {cjk_results['bleu4'].mean():.2f}")
        print(f"  Average chrF:   {cjk_results['chrf'].mean():.2f}")
        print(f"  Average CIDEr:  {cjk_results['cider'].mean():.2f}")

    # Print non-CJK averages
    non_cjk_results = results_df[~results_df['language'].isin(CJK_LANGUAGES)]
    if not non_cjk_results.empty:
        print(f"\nAverage Scores for Non-CJK Languages - {NUM_FEW_SHOT}-shot:")
        print(f"  Average BLEU-4: {non_cjk_results['bleu4'].mean():.2f}")
        print(f"  Average chrF:   {non_cjk_results['chrf'].mean():.2f}")
        print(f"  Average CIDEr:  {non_cjk_results['cider'].mean():.2f}")

    # Estimate costs
    total_requests = sum([len(preds) for preds in all_predictions.values()])
    # GPT-4o vision pricing is more complex, rough estimate
    print(f"\n💰 Cost Estimate:")
    print(f"  Total requests: {total_requests}")
    print(f"  Images per request: {NUM_FEW_SHOT + 1} (few-shot examples + target)")
    print(f"  Total images processed: {total_requests * (NUM_FEW_SHOT + 1)}")
    print(f"  Estimated cost (gpt-4o): ${total_requests * 0.01:.2f} - ${total_requests * 0.03:.2f}")
    print(f"  Note: Cost will be higher due to multiple images per request.")
    print(f"  Check platform.openai.com for exact billing")