from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict

DATASET = "tharindu/XL-MUNIChus"

# For CJK and special tokenization
try:
    import jieba  # Chinese
    import MeCab  # Japanese

    JIEBA_AVAILABLE = True
    MECAB_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    MECAB_AVAILABLE = False
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")

try:
    from pythainlp.tokenize import word_tokenize as thai_tokenize

    THAI_AVAILABLE = True
except ImportError:
    THAI_AVAILABLE = False
    print("Warning: pythainlp not installed for Thai tokenization.")
    print("  pip install pythainlp")

try:
    from pyvi import ViTokenizer

    VIETNAMESE_AVAILABLE = True
except ImportError:
    VIETNAMESE_AVAILABLE = False
    print("Warning: pyvi not installed for Vietnamese tokenization.")
    print("  pip install pyvi")

try:
    from kiwipiepy import Kiwi

    KOREAN_AVAILABLE = True
    korean_tokenizer = None  # Lazy initialization
except ImportError:
    KOREAN_AVAILABLE = False
    korean_tokenizer = None
    print("Warning: kiwipiepy not installed for Korean tokenization.")
    print("  pip install kiwipiepy")

try:
    from laonlp.tokenize import word_tokenize as lao_tokenize

    LAO_AVAILABLE = True
except ImportError:
    LAO_AVAILABLE = False
    print("Warning: laonlp not installed for Lao tokenization.")
    print("  pip install laonlp")

try:
    from khmernltk import word_tokenize as khmer_tokenize

    KHMER_AVAILABLE = True
except ImportError:
    KHMER_AVAILABLE = False
    print("Warning: khmernltk not installed for Khmer tokenization.")
    print("  pip install khmer-nltk")

try:
    from pyidaungsu import tokenize as myanmar_tokenize

    MYANMAR_AVAILABLE = True
except ImportError:
    MYANMAR_AVAILABLE = False
    print("Warning: pyidaungsu not installed for Burmese tokenization.")
    print("  pip install pyidaungsu")

try:
    import botok

    TIBETAN_AVAILABLE = True
    tibetan_tokenizer = None  # Lazy initialization
except ImportError:
    TIBETAN_AVAILABLE = False
    tibetan_tokenizer = None
    print("Warning: botok not installed for Tibetan tokenization.")
    print("  pip install botok")

# XL-MUNIChus language codes (ISO 639-3)
languages = [
    "amh", "ara", "bam", "ben", "bod", "bos", "bul", "ckb", "cmn", "cym",
    "deu", "eng", "fas", "fra", "gla", "guj", "hat", "hau", "hin", "hrv",
    "hye", "ibo", "ind", "jpn", "kat", "khm", "kin", "kir", "kmr", "kor",
    "lao", "lin", "mar", "mkd", "mya", "nde", "nep", "orm", "pan", "pcm",
    "pol", "prs", "pus", "ron", "rus", "sin", "sna", "som", "sqi", "srp",
    "swa", "tel", "tha", "tir", "tur", "ukr", "urd", "uzb", "vie", "yor"
]

language_names = {
    "amh": "Amharic",
    "ara": "Arabic",
    "bam": "Bambara",
    "ben": "Bengali",
    "bod": "Tibetan",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "deu": "German",
    "eng": "English",
    "fas": "Persian",
    "fra": "French",
    "gla": "Scottish Gaelic",
    "guj": "Gujarati",
    "hat": "Haitian Creole",
    "hau": "Hausa",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "jpn": "Japanese",
    "kat": "Georgian",
    "khm": "Khmer",
    "kin": "Kinyarwanda",
    "kir": "Kyrgyz",
    "kmr": "Kurmanji Kurdish",
    "kor": "Korean",
    "lao": "Lao",
    "lin": "Lingala",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mya": "Burmese",
    "nde": "North Ndebele",
    "nep": "Nepali",
    "orm": "Oromo",
    "pan": "Punjabi",
    "pcm": "Nigerian Pidgin",
    "pol": "Polish",
    "prs": "Dari",
    "pus": "Pashto",
    "ron": "Romanian",
    "rus": "Russian",
    "sin": "Sinhala",
    "sna": "Shona",
    "som": "Somali",
    "sqi": "Albanian",
    "srp": "Serbian",
    "swa": "Swahili",
    "tel": "Telugu",
    "tha": "Thai",
    "tir": "Tigrinya",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzb": "Uzbek",
    "vie": "Vietnamese",
    "yor": "Yoruba"
}

# Languages requiring special tokenization (no whitespace word boundaries)
SPECIAL_TOKENIZATION_LANGUAGES = {
    "cmn": "chinese",
    "jpn": "japanese",
    "kor": "korean",
    "tha": "thai",
    "vie": "vietnamese",
    "lao": "lao",
    "khm": "khmer",
    "mya": "burmese",
    "bod": "tibetan"
}


def tokenize_text(text, lang_code):
    """Tokenize text appropriately based on language"""
    global korean_tokenizer, tibetan_tokenizer

    if lang_code not in SPECIAL_TOKENIZATION_LANGUAGES:
        return text

    lang_type = SPECIAL_TOKENIZATION_LANGUAGES[lang_code]

    try:
        if lang_type == "chinese" and JIEBA_AVAILABLE:
            return " ".join(jieba.cut(text))

        elif lang_type == "japanese" and MECAB_AVAILABLE:
            mecab = MeCab.Tagger("-Owakati")
            return mecab.parse(text).strip()

        elif lang_type == "korean" and KOREAN_AVAILABLE:
            try:
                if korean_tokenizer is None:
                    from kiwipiepy import Kiwi
                    korean_tokenizer = Kiwi()
                tokens = korean_tokenizer.tokenize(text)
                return " ".join([token.form for token in tokens])
            except Exception as e:
                print(f"Korean tokenization failed: {e}, using character-level")
                return " ".join(list(text.replace(" ", "")))

        elif lang_type == "thai" and THAI_AVAILABLE:
            tokens = thai_tokenize(text, engine="newmm")
            return " ".join(tokens)

        elif lang_type == "vietnamese" and VIETNAMESE_AVAILABLE:
            return ViTokenizer.tokenize(text)

        elif lang_type == "lao" and LAO_AVAILABLE:
            tokens = lao_tokenize(text)
            return " ".join(tokens)

        elif lang_type == "khmer" and KHMER_AVAILABLE:
            tokens = khmer_tokenize(text)
            return " ".join(tokens)

        elif lang_type == "burmese" and MYANMAR_AVAILABLE:
            tokens = myanmar_tokenize(text, form="word")
            return " ".join(tokens)

        elif lang_type == "tibetan" and TIBETAN_AVAILABLE:
            try:
                if tibetan_tokenizer is None:
                    from botok import WordTokenizer
                    tibetan_tokenizer = WordTokenizer()
                tokens = tibetan_tokenizer.tokenize(text, split_affixes=False)
                return " ".join([t.text for t in tokens])
            except Exception as e:
                print(f"Tibetan tokenization failed: {e}, using character-level")
                return " ".join(list(text.replace(" ", "")))

        else:
            # Fallback to character-level tokenization
            return " ".join(list(text.replace(" ", "")))

    except Exception as e:
        print(f"Tokenization failed for {lang_code}: {e}, using character-level")
        return " ".join(list(text.replace(" ", "")))


def count_tokens(text, lang_code):
    """Count tokens in text based on language"""
    if not text:
        return 0

    text = tokenize_text(text, lang_code)

    return len(text.split())


def analyze_language(lang_code, dataset_name=DATASET):
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
        stats = analyze_language(lang, dataset_name=DATASET)

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

# STL (Special tokenisation languages)-specific averages
stl_stats = stats_df[stats_df['language'].isin(SPECIAL_TOKENIZATION_LANGUAGES.keys())]
if not stl_stats.empty:
    print("\n" + "=" * 80)
    print("CJK LANGUAGES AVERAGES (ja, zh, yue)")
    print("=" * 80)
    print(f"Average content tokens: {stl_stats['avg_content_tokens'].mean():.2f}")
    print(f"Average caption tokens: {stl_stats['avg_caption_tokens'].mean():.2f}")
    print(f"Average title tokens: {stl_stats['avg_title_tokens'].mean():.2f}")
    print(f"Average images per article: {stl_stats['avg_images_per_article'].mean():.2f}")

# Non-STL averages
non_stl_stats = stats_df[~stats_df['language'].isin(SPECIAL_TOKENIZATION_LANGUAGES.keys())]
if not non_stl_stats.empty:
    print("\n" + "=" * 80)
    print("NON-CJK LANGUAGES AVERAGES")
    print("=" * 80)
    print(f"Average content tokens: {non_stl_stats['avg_content_tokens'].mean():.2f}")
    print(f"Average caption tokens: {non_stl_stats['avg_caption_tokens'].mean():.2f}")
    print(f"Average title tokens: {non_stl_stats['avg_title_tokens'].mean():.2f}")
    print(f"Average images per article: {non_stl_stats['avg_images_per_article'].mean():.2f}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)