import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider

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

# Initialize model and processor
print("Loading Qwen3-VL-8B-Instruct...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Optional: Enable flash attention for better performance
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

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


def generate_caption_qwen3(image, news_content, language):
    """Generate caption using Qwen3-VL"""

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

    # Format message for Qwen3-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        # Apply chat template and prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        caption = output_text[0].strip()
        return caption

    except Exception as e:
        print(f"Error generating caption: {e}")
        import traceback
        traceback.print_exc()
        return ""


def evaluate_language(lang_code, dataset_name="tharindu/XL-MUNIChus", num_samples=None):
    """Evaluate Qwen3-VL model on a specific language with BLEU-4, CIDEr, and chrF"""

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
            generated_caption = generate_caption_qwen3(
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

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])

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

    for lang in languages:
        try:
            results, preds, refs = evaluate_language(
                lang,
                dataset_name="tharindu/XL-MUNIChus",
                num_samples=None  # Set to 10-100 for quick testing
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
    print("FINAL RESULTS SUMMARY - QWEN3-VL-8B-INSTRUCT ON XL-MUNIChus")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv("xl_munichus_qwen3_vl_results.csv", index=False)
    print("\n✓ Results saved to xl_munichus_qwen3_vl_results.csv")

    # Save detailed predictions
    with open("xl_munichus_qwen3_vl_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references
        }, f, ensure_ascii=False, indent=2)
    print("✓ Predictions saved to xl_munichus_qwen3_vl_predictions.json")

    # Overall averages
    print("\n" + "=" * 80)
    print("OVERALL AVERAGE SCORES")
    print("=" * 80)
    print(f"Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"Average CIDEr:  {results_df['cider'].mean():.2f}")