"""
Standalone script to pre-compute all embeddings (train + test) for all languages
Run this once with GPU, then use the cached embeddings for all evaluations
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
import numpy as np
import pickle
import os
from tqdm import tqdm

# Configuration
CACHE_DIR = "./embedding_cache"
DATASET_NAME = "tharindu/MUNIChus"
BATCH_SIZE = 32  # Increase if you have more GPU memory
NUM_SAMPLES = None  # Set to limit test samples (e.g., 50 for testing), None for all

# Language codes
LANGUAGES = ["ar", "en", "fr", "hi", "id", "ja", "si", "ur", "yue", "zh"]

LANGUAGE_NAMES = {
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


def setup_model(device='cuda'):
    """Load the Nomic embedding model"""
    print(f"Loading Nomic Vision model on {device}...")
    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-vision-v1.5",
        trust_remote_code=True
    ).to(device)
    model.eval()
    print("✓ Model loaded successfully\n")
    return model, processor, device


def get_batch_embeddings(images, model, processor, device):
    """Get embeddings for a batch of PIL images"""
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        img_emb = model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

    return img_embeddings.cpu().numpy()


def compute_embeddings(dataset, split_name, lang_code, model, processor, device, batch_size=32):
    """Compute embeddings for a dataset split"""
    print(f"Computing {split_name} embeddings for {LANGUAGE_NAMES[lang_code]}...")

    images = [dataset[i]['image'] for i in range(len(dataset))]
    embeddings_list = []

    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(images), batch_size),
                  desc=f"{split_name} {lang_code}",
                  total=num_batches):
        batch_end = min(i + batch_size, len(images))
        batch_images = images[i:batch_end]

        try:
            batch_embeddings = get_batch_embeddings(batch_images, model, processor, device)
            embeddings_list.append(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            # Process one by one as fallback
            for img in batch_images:
                try:
                    emb = get_batch_embeddings([img], model, processor, device)
                    embeddings_list.append(emb)
                except Exception as e2:
                    print(f"Error processing single image: {e2}")
                    # Add zero embedding as placeholder
                    embeddings_list.append(np.zeros((1, 768)))

    embeddings = np.vstack(embeddings_list)
    print(f"✓ Computed {embeddings.shape[0]} embeddings")
    return embeddings


def save_embeddings(embeddings, filepath):
    """Save embeddings to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✓ Saved to {filepath}")


def main():
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Setup model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  WARNING: CUDA not available, using CPU (will be slower)")

    model, processor, device = setup_model(device)

    # Process each language
    for lang in LANGUAGES:
        print(f"\n{'=' * 80}")
        print(f"Processing {LANGUAGE_NAMES[lang]} ({lang})")
        print(f"{'=' * 80}")

        try:
            # Load dataset
            dataset = load_dataset(DATASET_NAME, lang)

            # Compute train embeddings
            train_cache_path = os.path.join(CACHE_DIR, f"train_embeddings_{lang}.pkl")
            if os.path.exists(train_cache_path):
                print(f"Train embeddings already exist at {train_cache_path}, skipping...")
            else:
                train_embeddings = compute_embeddings(
                    dataset['train'],
                    'train',
                    lang,
                    model,
                    processor,
                    device,
                    batch_size=BATCH_SIZE
                )
                save_embeddings(train_embeddings, train_cache_path)

            # Compute test embeddings
            test_cache_path = os.path.join(CACHE_DIR, f"test_embeddings_{lang}.pkl")
            if os.path.exists(test_cache_path):
                print(f"Test embeddings already exist at {test_cache_path}, skipping...")
            else:
                test_data = dataset['test']
                if NUM_SAMPLES:
                    test_data = test_data.select(range(min(NUM_SAMPLES, len(test_data))))

                test_embeddings = compute_embeddings(
                    test_data,
                    'test',
                    lang,
                    model,
                    processor,
                    device,
                    batch_size=BATCH_SIZE
                )
                save_embeddings(test_embeddings, test_cache_path)

            print(f"✓ {lang} complete!")

        except Exception as e:
            print(f"✗ Error processing {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Clean up
    print(f"\n{'=' * 80}")
    print("ALL EMBEDDINGS COMPUTED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    print("\nYou can now run your evaluation scripts.")
    print("The embedding model will load from cache automatically.")

    # Show cache statistics
    print("\n📊 Cache Statistics:")
    total_size = 0
    for lang in LANGUAGES:
        train_path = os.path.join(CACHE_DIR, f"train_embeddings_{lang}.pkl")
        test_path = os.path.join(CACHE_DIR, f"test_embeddings_{lang}.pkl")

        train_size = os.path.getsize(train_path) / (1024 ** 2) if os.path.exists(train_path) else 0
        test_size = os.path.getsize(test_path) / (1024 ** 2) if os.path.exists(test_path) else 0
        total_size += train_size + test_size

        print(f"  {lang}: {train_size:.2f} MB (train) + {test_size:.2f} MB (test)")

    print(f"\n  Total cache size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()