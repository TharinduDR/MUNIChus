import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# Create output directory
output_dir = "similar_images_output"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("tharindu/MUNIChus", "en")
train_data = dataset['train']
test_data = dataset['test']

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Load vision encoder - USE FLOAT32 instead of FLOAT16
print("Loading nomic-embed-vision-v1.5...")
model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-vision-v1.5",
    trust_remote_code=True,
    torch_dtype=torch.float32
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

        embeddings.append(batch_embeddings.float().cpu().numpy())
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

# Normalize embeddings
print("\nNormalizing embeddings...")
with tqdm(total=2, desc="Normalizing", unit="step") as pbar:
    train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    pbar.update(1)
    test_embedding_norm = test_embedding / np.linalg.norm(test_embedding, axis=1, keepdims=True)
    pbar.update(1)

# Compute cosine similarities ONLY for this one test image
print("\nComputing similarities for selected test image...")
with tqdm(total=1, desc="Computing similarities", unit="step") as pbar:
    similarities = np.dot(test_embedding_norm, train_embeddings_norm.T)[0]
    pbar.update(1)

print(f"Similarities shape: {similarities.shape}")

# Get top-3 most similar images
print("\nFinding top-3 most similar images...")
with tqdm(total=1, desc="Finding top-3", unit="step") as pbar:
    top3_indices = np.argsort(similarities)[-3:][::-1]
    top3_similarities = similarities[top3_indices]
    pbar.update(1)

print(f"\nTop-3 indices: {top3_indices}")
print(f"Top-3 similarities: {top3_similarities}")

# Save individual images
print("\nSaving individual images...")

# Save test image
test_img = test_data[test_idx]['image']
test_caption = test_data[test_idx]['caption']

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(test_img)
ax.set_title(f'Test Image {test_idx}\nSimilarity: 1.000', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{output_dir}/test_image_{test_idx}.png', dpi=300, bbox_inches='tight')
plt.close()

# Save test image caption
with open(f'{output_dir}/test_image_{test_idx}_caption.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test Image Index: {test_idx}\n")
    f.write(f"Similarity: 1.000\n\n")
    f.write(f"Caption:\n{test_caption}")

print(f"  ✓ Saved: test_image_{test_idx}.png")
print(f"  ✓ Saved: test_image_{test_idx}_caption.txt")

# Save top-3 similar images
for i in range(3):
    similar_idx = int(top3_indices[i])
    similarity = float(top3_similarities[i])

    similar_img = train_data[similar_idx]['image']
    similar_caption = train_data[similar_idx]['caption']

    # Save image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(similar_img)
    ax.set_title(f'Similar Image {i + 1} (Train Index: {similar_idx})\nSimilarity: {similarity:.4f}',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similar_{i + 1}_train_{similar_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save caption
    with open(f'{output_dir}/similar_{i + 1}_train_{similar_idx}_caption.txt', 'w', encoding='utf-8') as f:
        f.write(f"Similar Image {i + 1}\n")
        f.write(f"Training Image Index: {similar_idx}\n")
        f.write(f"Similarity Score: {similarity:.4f}\n\n")
        f.write(f"Caption:\n{similar_caption}")

    print(f"  ✓ Saved: similar_{i + 1}_train_{similar_idx}.png")
    print(f"  ✓ Saved: similar_{i + 1}_train_{similar_idx}_caption.txt")

# Visualize all together with separate captions
print("\nCreating combined visualization...")
fig = plt.figure(figsize=(18, 10))

# Create a grid: 2 rows (images, captions) x 4 columns (test + 3 similar)
gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.3, wspace=0.2)

# Test image
ax0_img = fig.add_subplot(gs[0, 0])
ax0_img.imshow(test_img)
ax0_img.set_title(f'Test Image {test_idx}\nSimilarity: 1.000', fontsize=12, fontweight='bold')
ax0_img.axis('off')

ax0_cap = fig.add_subplot(gs[1, 0])
ax0_cap.text(0.5, 0.5, test_caption, fontsize=9, ha='center', va='center', wrap=True)
ax0_cap.axis('off')

# Top-3 similar images
for col in range(3):
    similar_idx = int(top3_indices[col])
    similarity = float(top3_similarities[col])

    similar_img = train_data[similar_idx]['image']
    similar_caption = train_data[similar_idx]['caption']

    # Image
    ax_img = fig.add_subplot(gs[0, col + 1])
    ax_img.imshow(similar_img)
    ax_img.set_title(f'Similar {col + 1}\nSimilarity: {similarity:.3f}', fontsize=12)
    ax_img.axis('off')

    # Caption
    ax_cap = fig.add_subplot(gs[1, col + 1])
    ax_cap.text(0.5, 0.5, similar_caption, fontsize=9, ha='center', va='center', wrap=True)
    ax_cap.axis('off')

plt.suptitle('Test Image with Top-3 Most Similar Training Images', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{output_dir}/combined_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: combined_visualization.png")

# Save summary text file
with open(f'{output_dir}/summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEST IMAGE\n")
    f.write("=" * 80 + "\n")
    f.write(f"Index: {test_idx}\n")
    f.write(f"Caption: {test_caption}\n\n")

    f.write("=" * 80 + "\n")
    f.write("TOP-3 SIMILAR IMAGES\n")
    f.write("=" * 80 + "\n")
    for i in range(3):
        similar_idx = int(top3_indices[i])
        similarity = float(top3_similarities[i])
        similar_caption = train_data[similar_idx]['caption']
        f.write(f"\n{i + 1}. Training Image {similar_idx}\n")
        f.write(f"   Similarity: {similarity:.4f}\n")
        f.write(f"   Caption: {similar_caption}\n")

print(f"  ✓ Saved: summary.txt")

# Also print the captions to console
print("\n" + "=" * 80)
print("TEST IMAGE")
print("=" * 80)
print(f"Index: {test_idx}")
print(f"Caption: {test_caption}")

print("\n" + "=" * 80)
print("TOP-3 SIMILAR IMAGES")
print("=" * 80)
for i in range(3):
    similar_idx = int(top3_indices[i])
    similarity = float(top3_similarities[i])
    similar_caption = train_data[similar_idx]['caption']
    print(f"\n{i + 1}. Training Image {similar_idx}")
    print(f"   Similarity: {similarity:.4f}")
    print(f"   Caption: {similar_caption}")

print(f"\n✓ All files saved to '{output_dir}/' directory")
print("\nGenerated files:")
print(f"  - test_image_{test_idx}.png")
print(f"  - test_image_{test_idx}_caption.txt")
for i in range(3):
    similar_idx = int(top3_indices[i])
    print(f"  - similar_{i + 1}_train_{similar_idx}.png")
    print(f"  - similar_{i + 1}_train_{similar_idx}_caption.txt")
print(f"  - combined_visualization.png")
print(f"  - summary.txt")