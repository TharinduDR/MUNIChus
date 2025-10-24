import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
dataset = load_dataset("tharindu/MUNIChus", "en")
train_data = dataset['train']
test_data = dataset['test']

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Load vision encoder
print("Loading nomic-embed-vision-v1.5...")
model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-vision-v1.5",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()

processor = AutoImageProcessor.from_pretrained(
    "nomic-ai/nomic-embed-vision-v1.5",
    trust_remote_code=True
)


def encode_images_batch(images, batch_size=32, desc="Encoding"):
    """Encode images in batches for much faster processing"""
    embeddings = []

    # Create progress bar
    pbar = tqdm(total=len(images), desc=desc, unit="img")

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(model.device)

        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1)

        embeddings.append(batch_embeddings.cpu().numpy())

        # Update progress bar
        pbar.update(len(batch))

    pbar.close()
    return np.vstack(embeddings)


# Select a random test image FIRST
np.random.seed(42)
test_idx = np.random.choice(len(test_data))

print(f"\nSelected test image index: {test_idx}")

# Only encode the ONE test image we need
print("\nEncoding selected test image...")
test_image = test_data[test_idx]['image']
test_embedding = encode_images_batch([test_image], batch_size=1, desc="Encoding test image")

# Encode all training images
print("\nEncoding training images...")
train_images = [example['image'] for example in train_data]
train_embeddings = encode_images_batch(train_images, batch_size=32, desc="Encoding train images")

print(f"\nTrain embeddings shape: {train_embeddings.shape}")
print(f"Test embedding shape: {test_embedding.shape}")

# Debug: Check embedding values
print(f"\nDEBUG - Test embedding stats:")
print(f"  Mean: {test_embedding.mean():.6f}")
print(f"  Std: {test_embedding.std():.6f}")
print(f"  Min: {test_embedding.min():.6f}")
print(f"  Max: {test_embedding.max():.6f}")
print(f"  Norm: {np.linalg.norm(test_embedding):.6f}")

print(f"\nDEBUG - Train embeddings stats (first sample):")
print(f"  Mean: {train_embeddings[0].mean():.6f}")
print(f"  Std: {train_embeddings[0].std():.6f}")
print(f"  Min: {train_embeddings[0].min():.6f}")
print(f"  Max: {train_embeddings[0].max():.6f}")
print(f"  Norm: {np.linalg.norm(train_embeddings[0]):.6f}")

# Normalize embeddings with progress bar
print("\nNormalizing embeddings...")
with tqdm(total=2, desc="Normalizing", unit="step") as pbar:
    train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    pbar.update(1)
    test_embedding_norm = test_embedding / np.linalg.norm(test_embedding, axis=1, keepdims=True)
    pbar.update(1)

print(f"\nDEBUG - After normalization:")
print(f"  Test embedding norm: {np.linalg.norm(test_embedding_norm):.6f}")
print(f"  Train embedding[0] norm: {np.linalg.norm(train_embeddings_norm[0]):.6f}")

# Compute cosine similarities ONLY for this one test image
print("\nComputing similarities for selected test image...")
with tqdm(total=1, desc="Computing similarities", unit="step") as pbar:
    similarities = np.dot(test_embedding_norm, train_embeddings_norm.T)[0]  # Shape: (num_train,)
    pbar.update(1)

print(f"Similarities shape: {similarities.shape}")
print(f"\nDEBUG - Similarity stats:")
print(f"  Mean: {similarities.mean():.6f}")
print(f"  Std: {similarities.std():.6f}")
print(f"  Min: {similarities.min():.6f}")
print(f"  Max: {similarities.max():.6f}")
print(f"  First 10 similarities: {similarities[:10]}")

# Get top-3 most similar images
print("\nFinding top-3 most similar images...")
with tqdm(total=1, desc="Finding top-3", unit="step") as pbar:
    top3_indices = np.argsort(similarities)[-3:][::-1]
    top3_similarities = similarities[top3_indices]
    pbar.update(1)

print(f"\nTop-3 indices: {top3_indices}")
print(f"Top-3 similarities: {top3_similarities}")

# Visualize
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Test image
test_img = test_data[test_idx]['image']
test_caption = test_data[test_idx]['caption']

axes[0].imshow(test_img)
axes[0].set_title(f'Test Image {test_idx}\nSimilarity: 1.000', fontsize=12, fontweight='bold')
axes[0].axis('off')
axes[0].text(0.5, -0.1, f"Caption: {test_caption[:80]}...",
             transform=axes[0].transAxes, fontsize=9, ha='center', wrap=True)

# Top-3 similar images
for col in range(3):
    similar_idx = int(top3_indices[col])
    similarity = float(top3_similarities[col])

    similar_img = train_data[similar_idx]['image']
    similar_caption = train_data[similar_idx]['caption']

    axes[col + 1].imshow(similar_img)
    axes[col + 1].set_title(f'Similar {col + 1}\nSimilarity: {similarity:.3f}', fontsize=12)
    axes[col + 1].axis('off')
    axes[col + 1].text(0.5, -0.1, f"Caption: {similar_caption[:80]}...",
                       transform=axes[col + 1].transAxes, fontsize=9, ha='center', wrap=True)

plt.suptitle('Test Image with Top-3 Most Similar Training Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('similar_images.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as 'similar_images.png'")
plt.show()