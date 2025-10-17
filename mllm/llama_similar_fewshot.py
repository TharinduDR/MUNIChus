import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
import numpy as np

"""
Similarity-based few-shot selection using Nomic Vision Embeddings
This module can be imported into any of the evaluation scripts
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import numpy as np
from PIL import Image
import pickle
import os
from tqdm import tqdm


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


def get_similarity_based_examples(test_image, train_dataset, lang_code,
                                  num_examples=3, selector=None):
    """
    Convenience function to get similarity-based few-shot examples

    Args:
        test_image: PIL Image (test image)
        train_dataset: HuggingFace dataset object (train split)
        lang_code: Language code
        num_examples: Number of examples to retrieve
        selector: SimilarityFewShotSelector instance (optional, will create if None)

    Returns:
        List of example dicts
    """
    if selector is None:
        selector = SimilarityFewShotSelector()

    return selector.find_similar_examples(
        test_image,
        train_dataset,
        lang_code,
        num_examples=num_examples
    )


# Example usage for integration into existing scripts:
"""
# At the top of your evaluation script:
from similarity_fewshot import SimilarityFewShotSelector

# Initialize once (loads model)
similarity_selector = SimilarityFewShotSelector(
    cache_dir="./embedding_cache",
    device='cuda'  # or 'cpu'
)

# In your evaluation function, before the test loop:
# Pre-compute embeddings for this language (done once, then cached)
similarity_selector.compute_train_embeddings(train_data, lang_code, batch_size=32)

# In your test loop, replace get_random_few_shot_examples with:
few_shot_examples = similarity_selector.find_similar_examples(
    test_image=example['image'],
    train_dataset=train_data,
    lang_code=lang_code,
    num_examples=num_few_shot
)

# The returned examples will have:
# - 'image': PIL Image
# - 'content': truncated news content
# - 'caption': reference caption
# - 'similarity': cosine similarity score (0-1)
# - 'index': index in training set
"""

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

# Load model
print("Loading Llama-3.2-11B-Vision-Instruct...")
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

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


def generate_caption_similarity_fewshot(image, news_content, language, few_shot_examples):
    """Generate caption using Llama-3.2-11B-Vision with similarity-based few-shot examples"""

    # Build few-shot prompt with examples
    prompt_parts = [
        f"You are an expert news image captioner for {language_names[language]} newspapers.",
        "\nHere are some examples of similar images with good captions:\n"
    ]

    # Add text-based few-shot examples with similarity scores
    for i, example in enumerate(few_shot_examples, 1):
        sim_score = example.get('similarity', 0)
        prompt_parts.append(f"Example {i} (similarity: {sim_score:.3f}):")
        prompt_parts.append(f"News context: {example['content']}")
        prompt_parts.append(f"Good caption: {example['caption']}\n")

    # Add the target task
    prompt_parts.append("\nNow, write a caption for a new image that is visually similar to the examples above.")
    prompt_parts.append(f"\nNews context: {news_content[:1200]}")
    prompt_parts.append(f"\nTask: Write a concise, informative caption for this image in {language_names[language]}.")
    prompt_parts.append("\nGuidelines:")
    prompt_parts.append(f"- Write in {language_names[language]} language only")
    prompt_parts.append("- Keep it brief (10-12 words)")
    prompt_parts.append("- Follow the style of the similar examples provided above")
    prompt_parts.append("- Identify key people, locations, and events in the image")
    prompt_parts.append("- Connect the visual content to the news context")
    prompt_parts.append("- Use journalistic style (factual, clear, objective)")
    prompt_parts.append(f"\nCaption in {language_names[language]}:")

    prompt = "\n".join(prompt_parts)

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

    # Extract caption
    if "assistant" in generated_text.lower():
        caption = generated_text.split("assistant")[-1].strip()
    elif language_names[language] in generated_text:
        parts = generated_text.split(language_names[language])
        caption = parts[-1].strip().lstrip(':').strip()
    else:
        caption = generated_text.split(prompt)[-1].strip()

    # Clean up any artifacts and common prefixes
    caption = caption.replace('</s>', '').replace('<s>', '').strip()

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


def evaluate_language_similarity_fewshot(lang_code, dataset_name="tharindu/MUNIChus",
                                         num_samples=None, num_few_shot=3,
                                         similarity_selector=None):
    """Evaluate Llama Vision model on a specific language with similarity-based few-shot learning"""

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

    predictions = []
    references = []
    similarity_scores_all = []  # Track similarity scores

    # Generate captions with similarity-based few-shot examples
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code} (similarity-based)")):
        try:
            # Use pre-computed test embedding instead of computing on-the-fly
            if hasattr(similarity_selector,
                       'test_embeddings_cache') and lang_code in similarity_selector.test_embeddings_cache:
                test_embedding = similarity_selector.test_embeddings_cache[lang_code][i]

                # Get train embeddings
                train_embeddings = similarity_selector.train_embeddings_cache[lang_code]

                # Compute similarities
                similarities = np.dot(train_embeddings, test_embedding)
                sorted_indices = np.argsort(similarities)[::-1]
                selected_indices = sorted_indices[:num_few_shot]

                # Build few-shot examples
                few_shot_examples = []
                for idx in selected_indices:
                    try:
                        train_example = train_data[int(idx)]
                        few_shot_examples.append({
                            'content': train_example['content'][:800],
                            'caption': train_example['caption'],
                            'similarity': float(similarities[idx]),
                            'index': int(idx)
                        })
                    except Exception as e:
                        print(f"Error loading example {idx}: {e}")
                        continue
            else:
                # Fallback to using the selector (requires model)
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
            generated_caption = generate_caption_similarity_fewshot(
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

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            similarity_scores_all.append([0] * num_few_shot)

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

    print(f"Starting {NUM_FEW_SHOT}-shot SIMILARITY-BASED evaluation for Llama-3.2-11B-Vision")
    print("Using Nomic Vision embeddings for similarity matching")
    print("=" * 80)

    # Initialize similarity selector (will cache embeddings)
    similarity_selector = SimilarityFewShotSelector(
        cache_dir="./embedding_cache",
        device='cpu'  # Use CPU to free GPU for Llama model
    )

    # Step 1: Pre-compute ALL embeddings (train + test) before deleting model
    print("Step 1: Pre-computing embeddings for all languages...")
    test_embeddings_cache = {}

    for lang in languages:
        train_data = load_train_dataset(lang)
        if train_data:
            # Compute train embeddings
            similarity_selector.compute_train_embeddings(train_data, lang, batch_size=16)

            # Compute test embeddings
            try:
                dataset = load_dataset("tharindu/MUNIChus", lang)
                test_data = dataset['test']
                if NUM_SAMPLES:
                    test_data = test_data.select(range(min(NUM_SAMPLES, len(test_data))))

                print(f"Computing test embeddings for {lang}...")
                test_images = [test_data[i]['image'] for i in range(len(test_data))]
                test_embeddings = []

                # Compute in batches
                batch_size = 16
                for i in tqdm(range(0, len(test_images), batch_size), desc=f"Test embeddings {lang}"):
                    batch = test_images[i:i + batch_size]
                    batch_emb = similarity_selector.get_batch_embeddings(batch)
                    test_embeddings.append(batch_emb)

                test_embeddings_cache[lang] = np.vstack(test_embeddings)
                print(f"✓ Cached {len(test_embeddings_cache[lang])} test embeddings for {lang}")

            except Exception as e:
                print(f"Error computing test embeddings for {lang}: {e}")

    # Delete the embedding model to free memory
    print("\n✓ All embeddings cached!")
    print("Freeing embedding model from memory...")
    del similarity_selector.vision_model
    del similarity_selector.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Memory freed!\n")

    # Store test embeddings in the selector for later use
    similarity_selector.test_embeddings_cache = test_embeddings_cache

    # Now run the main evaluation (using cached embeddings)
    print("Step 2: Running caption generation with cached embeddings...")
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

        except Exception as e:
            print(f"✗ Error evaluating {lang}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_results:
        print("\n✗ No results generated!")
    else:
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print(f"FINAL RESULTS SUMMARY - LLAMA 3.2 11B VISION (Similarity-based {NUM_FEW_SHOT}-shot)")
        print("=" * 80)
        print(results_df.to_string(index=False))

        output_prefix = f"llama_vision_{NUM_FEW_SHOT}shot_similarity"

        results_df.to_csv(f"{output_prefix}_evaluation_results.csv", index=False)
        print(f"\n✓ Results saved to {output_prefix}_evaluation_results.csv")

        with open(f"{output_prefix}_predictions.json", "w", encoding="utf-8") as f:
            json.dump({
                "predictions": all_predictions,
                "references": all_references,
                "config": {
                    "num_few_shot": NUM_FEW_SHOT,
                    "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
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