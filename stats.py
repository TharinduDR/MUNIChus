import cohere
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

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


def count_tokens(text, lang_code):
    """Count tokens in text based on language"""
    if not text:
        return 0

    # For CJK languages, use proper tokenization if available
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        if lang_code in ["zh", "yue"]:
            try:
                tokens = list(jieba.cut(text))
                return len(tokens)
            except Exception as e:
                # Fallback to character count
                return len(text.replace(" ", ""))

        elif lang_code == "ja":
            try:
                mecab = MeCab.Tagger("-Owakati")
                tokenized = mecab.parse(text).strip()
                return len(tokenized.split())
            except Exception as e:
                # Fallback to character count
                return len(text.replace(" ", ""))

    # For non-CJK languages, use whitespace tokenization
    return len(text.split())


def analyze_language(lang_code, dataset_name="tharindu/MUNIChus"):
    """Analyze statistics for a specific language"""

    print(f"\n{'=' * 80}")
    print(f"Analyzing {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    try:
        dataset = load_dataset(dataset_name, lang_code)

        # Combine train and test splits
        all_data = []
        if 'train' in dataset:
            train_data = dataset['train']
            all_data.extend(train_data)
            print(f"Loaded {len(train_data)} train examples")

        if 'test' in dataset:
            test_data = dataset['test']
            all_data.extend(test_data)
            print(f"Loaded {len(test_data)} test examples")

        print(f"Total examples: {len(all_data)}")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

    # Statistics collectors
    content_tokens = []
    caption_tokens = []
    title_tokens = []  # Changed from headline_tokens
    unique_contents = set()  # Track unique news content
    content_image_counts = defaultdict(int)  # Count images per unique content

    # Process each example
    for example in tqdm(all_data, desc=f"Processing {lang_code}"):
        try:
            # Get fields
            content = example.get('content', '')
            caption = example.get('caption', '')
            title = example.get('title', '')  # Changed from headline

            # Count tokens
            content_tokens.append(count_tokens(content, lang_code))
            caption_tokens.append(count_tokens(caption, lang_code))
            title_tokens.append(count_tokens(title, lang_code))  # Changed from headline_tokens

            # Track unique articles by content
            # Use content as identifier for unique articles
            if content:
                unique_contents.add(content)
                content_image_counts[content] += 1

        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    # Calculate statistics
    stats = {
        "language": lang_code,
        "language_name": language_names[lang_code],
        "total_samples": len(all_data),
        "unique_articles": len(unique_contents),
        "avg_content_tokens": sum(content_tokens) / len(content_tokens) if content_tokens else 0,
        "avg_caption_tokens": sum(caption_tokens) / len(caption_tokens) if caption_tokens else 0,
        "avg_title_tokens": sum(title_tokens) / len(title_tokens) if title_tokens else 0,
        "avg_images_per_article": len(all_data) / len(unique_contents) if unique_contents else 0,
        "min_content_tokens": min(content_tokens) if content_tokens else 0,
        "max_content_tokens": max(content_tokens) if content_tokens else 0,
        "min_caption_tokens": min(caption_tokens) if caption_tokens else 0,
        "max_caption_tokens": max(caption_tokens) if caption_tokens else 0,
    }

    # Print summary
    print(f"\nStatistics for {language_names[lang_code]}:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique articles: {stats['unique_articles']}")
    print(f"  Avg images per article: {stats['avg_images_per_article']:.2f}")
    print(f"  Avg content tokens: {stats['avg_content_tokens']:.2f}")
    print(f"  Avg caption tokens: {stats['avg_caption_tokens']:.2f}")
    print(f"  Avg title tokens: {stats['avg_title_tokens']:.2f}")

    return stats


# Analyze all languages
all_stats = []

print("Starting dataset analysis...")
print("=" * 80)

for lang in languages:
    try:
        stats = analyze_language(lang, dataset_name="tharindu/MUNIChus")

        if stats is not None:
            all_stats.append(stats)

    except Exception as e:
        print(f"✗ Error analyzing {lang}: {e}")
        import traceback

        traceback.print_exc()
        continue

if not all_stats:
    print("\n✗ No statistics generated!")
    import sys

    sys.exit(1)

# Create DataFrame and save results
stats_df = pd.DataFrame(all_stats)

print("\n" + "=" * 80)
print("DATASET STATISTICS SUMMARY")
print("=" * 80)
print(stats_df.to_string(index=False))

# Save to CSV
stats_df.to_csv("dataset_statistics.csv", index=False)
print("\n✓ Statistics saved to dataset_statistics.csv")

# Save detailed JSON
with open("dataset_statistics.json", "w", encoding="utf-8") as f:
    json.dump(all_stats, f, ensure_ascii=False, indent=2)
print("✓ Detailed statistics saved to dataset_statistics.json")

# Calculate overall averages
print("\n" + "=" * 80)
print("OVERALL AVERAGES ACROSS ALL LANGUAGES")
print("=" * 80)
print(f"Average content tokens: {stats_df['avg_content_tokens'].mean():.2f}")
print(f"Average caption tokens: {stats_df['avg_caption_tokens'].mean():.2f}")
print(f"Average title tokens: {stats_df['avg_title_tokens'].mean():.2f}")
print(f"Average images per article: {stats_df['avg_images_per_article'].mean():.2f}")
print(f"Total unique articles: {stats_df['unique_articles'].sum()}")
print(f"Total samples: {stats_df['total_samples'].sum()}")

# CJK-specific averages
cjk_stats = stats_df[stats_df['language'].isin(CJK_LANGUAGES)]
if not cjk_stats.empty:
    print("\n" + "=" * 80)
    print("CJK LANGUAGES AVERAGES (ja, zh, yue)")
    print("=" * 80)
    print(f"Average content tokens: {cjk_stats['avg_content_tokens'].mean():.2f}")
    print(f"Average caption tokens: {cjk_stats['avg_caption_tokens'].mean():.2f}")
    print(f"Average title tokens: {cjk_stats['avg_title_tokens'].mean():.2f}")
    print(f"Average images per article: {cjk_stats['avg_images_per_article'].mean():.2f}")

# Non-CJK averages
non_cjk_stats = stats_df[~stats_df['language'].isin(CJK_LANGUAGES)]
if not non_cjk_stats.empty:
    print("\n" + "=" * 80)
    print("NON-CJK LANGUAGES AVERAGES")
    print("=" * 80)
    print(f"Average content tokens: {non_cjk_stats['avg_content_tokens'].mean():.2f}")
    print(f"Average caption tokens: {non_cjk_stats['avg_caption_tokens'].mean():.2f}")
    print(f"Average title tokens: {non_cjk_stats['avg_title_tokens'].mean():.2f}")
    print(f"Average images per article: {non_cjk_stats['avg_images_per_article'].mean():.2f}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)