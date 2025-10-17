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
import numpy as np

# Import similarity selector
from similarity_fewshot import SimilarityFewShotSelector

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


def generate_caption_cohere_similarity_fewshot(image, news_content, language, few_shot_examples, max_retries=5):
    """Generate caption using Cohere with similarity-based few-shot examples"""

    # Convert target image to data URL
    image_url = image_to_data_url(image)
    if image_url is None:
        return ""

    # Prepare message content with all images and text
    message_content = []

    # Add few-shot example images and their context
    for i, example in enumerate(few_shot_examples, 1):
        example_image_url = image_to_data_url(example['image'])
        if example_image_url:
            message_content.append({
                "type": "text",
                "text": f"Example {i} (similarity: {example.get('similarity', 'N/A'):.3f}):"
            })
            message_content.append({
                "type": "image",
                "image": example_image_url
            })
            message_content.append({
                "type": "text",
                "text": f"News excerpt: {example['content']}\nCaption: {example['caption']}\n"
            })

    # Add the target image and prompt
    message_content.append({
        "type": "text",
        "text": "\nNow, write a caption for this new image (similar to the examples above):"
    })
    message_content.append({
        "type": "image",
        "image": image_url
    })

    # Create the final prompt
    prompt = f"""Given the news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {language_names[language]}.

Guidelines:
- Write in {language_names[language]} language only
- Keep it brief (10-12 words)
- Follow the style of the similar examples provided above
- Use journalistic style (factual, clear, objective)

Caption in {language_names[language]}:"""

    message_content.append({
        "type": "text",
        "text": prompt
    })

    # Call Cohere API with retries and exponential backoff
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model="c4ai-aya-vision-32b",
                messages=[
                    {
                        "role": "user",
                        "content": message_content
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
                wait_time = (2 ** attempt) * 2  # Exponential backoff
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


def evaluate_language_similarity_fewshot(lang_code, dataset_name="tharindu/MUNIChus",
                                        num_samples=None, num_few_shot=3,
                                        similarity_selector=None):
    """Evaluate Cohere model on a specific language with similarity-based few-shot learning"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code}) with {num_few_shot}-shot similarity-based learning")
    print(f"{'=' * 80}")

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

    # Pre-compute embeddings for training set (cached for efficiency)
    print(f"Pre-computing/loading embeddings for {lang_code} training set...")
    similarity_selector.compute_train_embeddings(train_data, lang_code)
    print(f"✓ Embeddings ready for {lang_code}")

    predictions = []
    references = []
    similarity_scores_all = []  # Track similarity scores

    # Generate captions with similarity-based few-shot examples
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code} (similarity-based)")):
        try:
            # Find similar examples from training set
            few_shot_examples = similarity_selector.find_similar_examples(
                test_image=example['image'],
                train_dataset=train_data,
                lang_code=lang_code,
                num_examples=num_few_shot
            )

            if len(few_shot_examples) < num_few_shot:
                print(f"Warning: Only {len(few_shot_examples)} similar examples found")

            # Track similarity scores
            sim_scores = [ex.get('similarity', 0) for ex in few_shot_examples]
            similarity_scores_all.append(sim_scores)

            # Generate caption with similar few-shot examples
            generated_caption = generate_caption_cohere_similarity_fewshot(
                example['image'],
                example['content'],
                lang_code,
                few_shot_examples
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            # Print first few examples with similarity info
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Using {len(few_shot_examples)} similar examples")
                print(f"Similarity scores: {[f'{s:.3f}' for s in sim_scores]}")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

            # Rate limiting
            time.sleep(12)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            similarity_scores_all.append([0] * num_few_shot)
            time.sleep(12)

    # Validate predictions
    valid_predictions = [p for p in predictions if p]
    if not valid_predictions:
        print(f"✗ No valid predictions for {lang_code}")
        return None, predictions, references

    print(f"\nGenerated {len(valid_predictions)}/{len(predictions)} valid captions")

    # Report average similarity scores
    if similarity_scores_all:
        avg_similarities = np.mean([np.mean(scores) for scores in similarity_scores_all if scores])
        print(f"Average similarity score: {avg_similarities:.3f}")

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
            "avg_similarity": float(avg_similarities) if similarity_scores_all else 0.0,
            "bleu4": bleu_score.score,
            "chrf": chrf_score.score,
            "cider": cider_score * 100
        }

        print(f"\nResults for {language_names[lang_code]} (similarity-based {num_few_shot}-shot):")
        print(f"  Valid captions: {len(valid_predictions)}/{len(predictions)}")
        print(f"  Avg similarity: {results['avg_similarity']:.3f}")
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

    print(f"Starting {NUM_FEW_SHOT}-shot SIMILARITY-BASED evaluation")
    print("Using Nomic Vision embeddings for similarity matching")
    print("=" * 80)

    # Initialize similarity selector (will cache embeddings)
    similarity_selector = SimilarityFewShotSelector(cache_dir="./embedding_cache")

    for lang in languages:
        try:
            results, preds, refs = evaluate_language_similarity_fewshot(
                lang,
                dataset_name="tharindu/MUNIChus",
                num_samples=NUM_SAMPLES,
                num_few_shot=NUM_FEW_SHOT,
                similarity_selector=similarity_selector
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
    print(f"FINAL RESULTS SUMMARY - COHERE Vision (Similarity-based {NUM_FEW_SHOT}-shot)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    output_prefix = f"cohere_vision_{NUM_FEW_SHOT}shot_similarity"

    results_df.to_csv(f"{output_prefix}_evaluation_results.csv", index=False)
    print(f"\n✓ Results saved to {output_prefix}_evaluation_results.csv")

    with open(f"{output_prefix}_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references,
            "config": {
                "num_few_shot": NUM_FEW_SHOT,
                "model": "c4ai-aya-vision-32b",
                "few_shot_method": "similarity-based",
                "embedding_model": "nomic-embed-vision-v1.5"
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Predictions saved to {output_prefix}_predictions.json")

    # Calculate averages
    print("\nAverage Scores Across All Languages:")
    print(f"  Average Similarity: {results_df['avg_similarity'].mean():.3f}")
    print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")

    # Print CJK-specific averages
    cjk_results = results_df[results_df['language'].isin(CJK_LANGUAGES)]
    if not cjk_results.empty:
        print(f"\nAverage Scores for CJK Languages (ja, zh, yue) - Similarity-based {NUM_FEW_SHOT}-shot:")
        print(f"  Average Similarity: {cjk_results['avg_similarity'].mean():.3f}")
        print(f"  Average BLEU-4: {cjk_results['bleu4'].mean():.2f}")
        print(f"  Average chrF:   {cjk_results['chrf'].mean():.2f}")
        print(f"  Average CIDEr:  {cjk_results['cider'].mean():.2f}")

    # Print non-CJK averages
    non_cjk_results = results_df[~results_df['language'].isin(CJK_LANGUAGES)]
    if not non_cjk_results.empty:
        print(f"\nAverage Scores for Non-CJK Languages - Similarity-based {NUM_FEW_SHOT}-shot:")
        print(f"  Average Similarity: {non_cjk_results['avg_similarity'].mean():.3f}")
        print(f"  Average BLEU-4: {non_cjk_results['bleu4'].mean():.2f}")
        print(f"  Average chrF:   {non_cjk_results['chrf'].mean():.2f}")
        print(f"  Average CIDEr:  {non_cjk_results['cider'].mean():.2f}")