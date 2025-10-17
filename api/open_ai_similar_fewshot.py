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
import numpy as np

"""
Similarity-based few-shot selection using Nomic Vision Embeddings
This module can be imported into any of the evaluation scripts
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import pickle
from PIL import Image


class SimilarityFewShotSelector:
    def __init__(self, cache_dir="./embedding_cache", device=None):
        """
        Initialize the similarity-based few-shot selector

        Args:
            cache_dir: Directory to cache embeddings to avoid recomputation
            device: Device to use for model ('cuda', 'cpu', or None for auto)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.train_embeddings_cache = {}

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model and processor
        print(f"Loading Nomic Vision model on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5",
            trust_remote_code=True
        ).to(self.device)
        self.vision_model.eval()  # Set to evaluation mode
        print("✓ Nomic Vision model loaded successfully")

    def get_image_embedding(self, image):
        """
        Get embedding for a single PIL image

        Args:
            image: PIL Image object

        Returns:
            numpy array of shape (768,) containing the normalized embedding
        """
        # Process image
        inputs = self.processor(image, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            img_emb = self.vision_model(**inputs).last_hidden_state
            img_embedding = F.normalize(img_emb[:, 0], p=2, dim=1)

        # Convert to numpy and return
        return img_embedding.cpu().numpy()[0]

    def get_batch_embeddings(self, images):
        """
        Get embeddings for a batch of PIL images

        Args:
            images: List of PIL Image objects

        Returns:
            numpy array of shape (batch_size, 768)
        """
        # Process batch of images
        inputs = self.processor(images, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            img_emb = self.vision_model(**inputs).last_hidden_state
            img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

        # Convert to numpy and return
        return img_embeddings.cpu().numpy()

    def compute_train_embeddings(self, train_dataset, lang_code, force_recompute=False, batch_size=32):
        """
        Compute and cache embeddings for all training images

        Args:
            train_dataset: HuggingFace dataset object (train split)
            lang_code: Language code (e.g., 'en', 'ja')
            force_recompute: If True, recompute even if cache exists
            batch_size: Batch size for processing images

        Returns:
            numpy array of shape (num_train_examples, 768)
        """
        cache_path = os.path.join(self.cache_dir, f"train_embeddings_{lang_code}.pkl")

        # Check if cached embeddings exist
        if not force_recompute and os.path.exists(cache_path):
            print(f"Loading cached embeddings for {lang_code} from {cache_path}")
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            self.train_embeddings_cache[lang_code] = embeddings
            print(f"✓ Loaded {embeddings.shape[0]} cached embeddings")
            return embeddings

        print(f"Computing embeddings for {len(train_dataset)} training images ({lang_code})...")
        print("This may take a while but will be cached for future use.")

        embeddings_list = []

        # Process in batches
        num_batches = (len(train_dataset) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(train_dataset), batch_size),
                      desc=f"Embedding {lang_code}",
                      total=num_batches):
            batch_end = min(i + batch_size, len(train_dataset))

            # Get batch of images
            batch_images = [train_dataset[j]['image'] for j in range(i, batch_end)]

            # Get embeddings for batch
            try:
                batch_embeddings = self.get_batch_embeddings(batch_images)
                embeddings_list.append(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {e}")
                # Fallback to processing one by one
                for img in batch_images:
                    try:
                        emb = self.get_image_embedding(img)
                        embeddings_list.append(emb.reshape(1, -1))
                    except Exception as e2:
                        print(f"Error processing single image: {e2}")
                        # Add zero embedding as placeholder
                        embeddings_list.append(np.zeros((1, 768)))

        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list)

        # Cache the embeddings
        print(f"Saving embeddings to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Cached {embeddings.shape[0]} embeddings to {cache_path}")

        self.train_embeddings_cache[lang_code] = embeddings
        return embeddings

    def find_similar_examples(self, test_image, train_dataset, lang_code,
                              num_examples=3, exclude_indices=None):
        """
        Find the most similar training examples for a test image

        Args:
            test_image: PIL Image (test image)
            train_dataset: HuggingFace dataset object (train split)
            lang_code: Language code
            num_examples: Number of similar examples to return
            exclude_indices: List of indices to exclude from selection

        Returns:
            List of dicts with keys: 'image', 'content', 'caption', 'similarity', 'index'
        """
        # Get or compute train embeddings
        if lang_code not in self.train_embeddings_cache:
            train_embeddings = self.compute_train_embeddings(train_dataset, lang_code)
        else:
            train_embeddings = self.train_embeddings_cache[lang_code]

        # Get embedding for test image
        test_embedding = self.get_image_embedding(test_image)

        # Compute cosine similarities (embeddings are already normalized)
        similarities = np.dot(train_embeddings, test_embedding)

        # Get indices sorted by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter out excluded indices
        if exclude_indices is not None:
            exclude_set = set(exclude_indices)
            sorted_indices = [idx for idx in sorted_indices if idx not in exclude_set]

        # Select top k
        selected_indices = sorted_indices[:num_examples]

        # Prepare examples
        examples = []
        for idx in selected_indices:
            try:
                example = train_dataset[int(idx)]
                examples.append({
                    'image': example['image'],
                    'content': example['content'][:800],
                    'caption': example['caption'],
                    'similarity': float(similarities[idx]),
                    'index': int(idx)
                })
            except Exception as e:
                print(f"Error loading example {idx}: {e}")
                continue

        return examples


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


def generate_caption_gpt4_similarity_fewshot(image, news_content, language, few_shot_examples, max_retries=3):
    """Generate caption using GPT-4 Vision with similarity-based few-shot examples"""

    # Encode target image to base64
    target_image_base64 = encode_image_to_base64(image)

    # Build message content with few-shot examples
    message_content = []

    # Add introduction
    message_content.append({
        "type": "text",
        "text": f"You are writing captions for newspaper images in {language_names[language]}.\n\nHere are some examples of good image captions from similar images:\n"
    })

    # Add few-shot examples with similarity scores
    for i, example in enumerate(few_shot_examples, 1):
        example_image_base64 = encode_image_to_base64(example['image'])

        message_content.append({
            "type": "text",
            "text": f"\nExample {i} (similarity: {example.get('similarity', 'N/A'):.3f}):"
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
        "text": f"\nNow, write a caption for this new image (similar to the examples above):"
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
- Follow the style of the similar examples provided above
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


def evaluate_language_similarity_fewshot(lang_code, dataset_name="tharindu/MUNIChus",
                                        num_samples=None, num_few_shot=3,
                                        similarity_selector=None):
    """Evaluate GPT-4 Vision model on a specific language with similarity-based few-shot learning"""

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
            generated_caption = generate_caption_gpt4_similarity_fewshot(
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
            time.sleep(1)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            similarity_scores_all.append([0] * num_few_shot)
            time.sleep(1)

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

    print(f"Starting {NUM_FEW_SHOT}-shot SIMILARITY-BASED evaluation with GPT-4 Vision")
    print("Using Nomic Vision embeddings for similarity matching")
    print("Note: This will incur API costs. Monitor your usage at platform.openai.com")
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
    print(f"FINAL RESULTS SUMMARY - GPT-4 VISION (Similarity-based {NUM_FEW_SHOT}-shot)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    output_prefix = f"gpt4_vision_{NUM_FEW_SHOT}shot_similarity"

    results_df.to_csv(f"{output_prefix}_evaluation_results.csv", index=False)
    print(f"\n✓ Results saved to {output_prefix}_evaluation_results.csv")

    with open(f"{output_prefix}_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references,
            "config": {
                "num_few_shot": NUM_FEW_SHOT,
                "model": "gpt-4o",
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

    # Estimate costs
    total_requests = sum([len(preds) for preds in all_predictions.values()])
    print(f"\n💰 Cost Estimate:")
    print(f"  Total requests: {total_requests}")
    print(f"  Images per request: {NUM_FEW_SHOT + 1} (similar examples + target)")
    print(f"  Total images processed: {total_requests * (NUM_FEW_SHOT + 1)}")
    print(f"  Estimated cost (gpt-4o): ${total_requests * 0.01:.2f} - ${total_requests * 0.03:.2f}")
    print(f"  Note: Cost will be higher due to multiple images per request.")
    print(f"  Check platform.openai.com for exact billing")