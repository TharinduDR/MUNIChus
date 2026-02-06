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
    import pyidaungsu as pds
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

# Initialize OpenAI client
print("Initializing OpenAI client...")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize metrics
chrf_metric = CHRF()
cider_scorer = Cider()

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
            tokens = pds.tokenize(text, form="word")
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


def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_caption_gpt4(image, news_content, language, max_retries=3):
    """Generate caption using GPT-4 Vision"""

    # Encode image to base64
    image_base64 = encode_image_to_base64(image)

    # Create prompt
    prompt = f"""You are writing a caption for a newspaper image.

Given the image and this news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {language_names[language]}.

Guidelines:
- Write in {language_names[language]} language only
- Keep it brief (1-2 sentences)
- Identify and include: people's names, locations, and organizations visible in the image
- Connect what you see in the image to the news context
- Use journalistic style (factual, clear, objective)
- Focus on the main subject of the image

Caption in {language_names[language]}:"""

    # Call OpenAI API with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )

            # Extract response
            caption = response.choices[0].message.content.strip()

            # Clean common prefixes
            prefixes = [
                f"Caption in {language_names[language]}:",
                "Caption:",
                "Here is the caption:",
                f"{language_names[language]} caption:"
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


def evaluate_language(lang_code, dataset_name="tharindu/XL-MUNIChus", num_samples=None):
    """Evaluate GPT-4 Vision model on a specific language with BLEU-4, CIDEr, and chrF"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    # Load test set
    try:
        dataset = load_dataset(dataset_name, lang_code)
        test_data = dataset['test']
    except Exception as e:
        print(f"Error loading dataset for {lang_code}: {e}")
        return None, None, None

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    print(f"Processing {len(test_data)} examples...")

    predictions = []
    references = []

    # Generate captions
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        try:
            generated_caption = generate_caption_gpt4(
                example['image'],
                example['content'],
                lang_code
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            # Print sample outputs
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

            # Rate limiting - adjust based on your tier
            time.sleep(0.5)  # 0.5 second between requests

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            time.sleep(0.5)

    # Tokenize for special languages if needed
    needs_tokenization = lang_code in SPECIAL_TOKENIZATION_LANGUAGES
    if needs_tokenization:
        print(f"Tokenizing texts for {lang_code} (special tokenization required)...")
        tokenized_predictions = [tokenize_text(pred, lang_code) for pred in predictions]
        tokenized_references = [[tokenize_text(ref[0], lang_code)] for ref in references]
    else:
        tokenized_predictions = predictions
        tokenized_references = references

    # Calculate BLEU-4 with tokenized texts
    print(f"\nCalculating BLEU-4 for {lang_code}...")
    if needs_tokenization:
        print(f"  Using proper tokenization for BLEU-4")

    bleu_metric = BLEU(max_ngram_order=4)
    tokenized_references_transposed = [[ref[0] for ref in tokenized_references]]
    bleu_score = bleu_metric.corpus_score(tokenized_predictions, tokenized_references_transposed)

    # Calculate chrF (character-level, no tokenization needed)
    print(f"Calculating chrF for {lang_code}...")
    references_transposed = [[ref[0] for ref in references]]
    chrf_score = chrf_metric.corpus_score(predictions, references_transposed)

    # Calculate CIDEr with tokenized texts
    print(f"Calculating CIDEr for {lang_code}...")
    if needs_tokenization:
        print(f"  Using proper tokenization for CIDEr")

    predictions_dict = {i: [pred] for i, pred in enumerate(tokenized_predictions)}
    references_dict = {i: refs for i, refs in enumerate(tokenized_references)}
    cider_score, _ = cider_scorer.compute_score(references_dict, predictions_dict)

    results = {
        "language": lang_code,
        "language_name": language_names[lang_code],
        "num_samples": len(predictions),
        "bleu4": bleu_score.score,
        "chrf": chrf_score.score,
        "cider": cider_score * 100
    }

    print(f"\nResults for {language_names[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references


# Main execution
if __name__ == "__main__":
    # Evaluate all languages
    all_results = []
    all_predictions = {}
    all_references = {}

    print("Starting evaluation with GPT-4 Vision on XL-MUNIChus...")
    print("Note: This will incur API costs. Monitor your usage at platform.openai.com")
    print("=" * 80)

    for lang in languages:
        try:
            results, preds, refs = evaluate_language(
                lang,
                dataset_name="tharindu/XL-MUNIChus",
                num_samples=None  # Set to 50-100 to control costs during testing
            )

            if results is not None:
                all_results.append(results)
                all_predictions[lang] = preds
                all_references[lang] = refs

        except Exception as e:
            print(f"Error evaluating {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY - GPT-4 VISION ON XL-MUNIChus")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv("xl_munichus_gpt4_vision_results.csv", index=False)
    print("\n✓ Results saved to xl_munichus_gpt4_vision_results.csv")

    # Save detailed predictions
    with open("xl_munichus_gpt4_vision_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references
        }, f, ensure_ascii=False, indent=2)
    print("✓ Predictions saved to xl_munichus_gpt4_vision_predictions.json")

    # Overall averages
    print("\n" + "=" * 80)
    print("OVERALL AVERAGE SCORES")
    print("=" * 80)
    print(f"Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"Average CIDEr:  {results_df['cider'].mean():.2f}")

    # Estimate costs
    total_requests = sum([len(preds) for preds in all_predictions.values()])
    print(f"\n💰 Cost Estimate:")
    print(f"  Total requests: {total_requests}")
    print(f"  Estimated cost (gpt-4o): ${total_requests * 0.005:.2f} - ${total_requests * 0.015:.2f}")
    print(f"  Note: Actual costs may vary. Check platform.openai.com for exact billing")