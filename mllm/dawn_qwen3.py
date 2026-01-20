import argparse
import os
import gc
import json
import signal
from functools import wraps

import torch
import intel_extension_for_pytorch as ipex
import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider


class TimeoutError(Exception):
    pass


def timeout(seconds):
    """Decorator to add timeout to a function"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds} seconds")

            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

set_seed(777)

# Language configuration
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

CJK_LANGUAGES = ["ja", "zh", "yue"]


def log_gpu_memory():
    """Log GPU memory usage for all available XPU devices"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("\n" + "=" * 60)
        print("GPU Memory Usage:")
        for i in range(torch.xpu.device_count()):
            allocated = torch.xpu.memory_allocated(i) / 1024 ** 3
            reserved = torch.xpu.memory_reserved(i) / 1024 ** 3
            print(f"  GPU {i}:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved:  {reserved:.2f} GB")
        print("=" * 60 + "\n")


def clear_memory():
    """Aggressively clear GPU memory with explicit synchronization"""
    gc.collect()
    if hasattr(torch, 'xpu'):
        for i in range(torch.xpu.device_count()):
            with torch.xpu.device(i):
                torch.xpu.synchronize()  # Wait for all operations to complete
                torch.xpu.empty_cache()
                torch.xpu.synchronize()  # Ensure cache is cleared


def sync_all_devices():
    """Force synchronization across all XPU devices"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            with torch.xpu.device(i):
                torch.xpu.synchronize()


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


def generate_caption(model, processor, image, news_content, language, timeout_seconds=120):
    """Generate caption using Qwen3-VL with timeout protection"""

    prompt = f"""You are writing a caption for a newspaper image.

Given the image and this news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {LANGUAGE_NAMES[language]}.

Guidelines:
- Write in {LANGUAGE_NAMES[language]} language only
- Keep it brief (10-12 words)
- Identify and include: people's names, locations, and organizations visible in the image
- Connect what you see in the image to the news context
- Use journalistic style (factual, clear, objective)
- Focus on the main subject of the image

Caption in {LANGUAGE_NAMES[language]}:"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    def _generate():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Explicit sync before generation
        sync_all_devices()

        generated_ids = model.generate(**inputs, max_new_tokens=100)

        # Explicit sync after generation
        sync_all_devices()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()

    try:
        # Set up timeout using signal
        def handler(signum, frame):
            raise TimeoutError(f"Generation timed out after {timeout_seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_seconds)

        try:
            caption = _generate()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return caption

    except TimeoutError as e:
        print(f"Timeout: {e}")
        clear_memory()
        return ""

    except RuntimeError as e:
        error_str = str(e)
        if "OUT_OF_RESOURCES" in error_str or "out of memory" in error_str.lower():
            print(f"OOM error, clearing cache and retrying with shorter generation...")
            clear_memory()

            try:
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)

                try:
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs = inputs.to(model.device)
                    sync_all_devices()

                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                    sync_all_devices()

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    return output_text[0].strip()
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
                clear_memory()
                return ""
        else:
            print(f"Error generating caption: {e}")
            clear_memory()
            return ""

    except Exception as e:
        print(f"Error generating caption: {e}")
        clear_memory()
        return ""


def evaluate_language(model, processor, lang_code, dataset_name, num_samples=None, output_folder=None):
    """Evaluate model on a specific language with BLEU-4, CIDEr, and chrF"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {LANGUAGE_NAMES[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    # Load test set
    dataset = load_dataset(dataset_name, lang_code)
    test_data = dataset['test']

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    print(f"Processing {len(test_data)} examples...")

    predictions = []
    references = []

    # Generate captions
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        # Clear memory more frequently (every 5 samples) and sync
        if i % 5 == 0:
            sync_all_devices()
            clear_memory()

        try:
            generated_caption = generate_caption(
                model,
                processor,
                example['image'],
                example['content'],
                lang_code,
                timeout_seconds=120  # 2 minute timeout per sample
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            # Sync after each successful generation
            sync_all_devices()

            # Print sample outputs
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

            # Save intermediate results every 100 samples
            if output_folder and (i + 1) % 100 == 0:
                intermediate_file = os.path.join(output_folder, f"predictions_{lang_code}_checkpoint_{i + 1}.json")
                with open(intermediate_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "predictions": predictions,
                        "references": [ref[0] for ref in references],
                        "completed": i + 1
                    }, f, ensure_ascii=False, indent=2)
                print(f"\nCheckpoint saved: {intermediate_file}")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])
            sync_all_devices()
            clear_memory()

    # Initialize metrics
    chrf_metric = CHRF()
    cider_scorer = Cider()

    # Tokenize for CJK languages if needed
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"Tokenizing texts for {lang_code} (CJK language)...")
        tokenized_predictions = [tokenize_text(pred, lang_code) for pred in predictions]
        tokenized_references = [[tokenize_text(ref[0], lang_code)] for ref in references]
    else:
        tokenized_predictions = predictions
        tokenized_references = references

    # Calculate BLEU-4
    print(f"Calculating BLEU-4 for {lang_code}...")
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
        "language_name": LANGUAGE_NAMES[lang_code],
        "num_samples": len(predictions),
        "bleu4": bleu_score.score,
        "chrf": chrf_score.score,
        "cider": cider_score * 100
    }

    print(f"\nResults for {LANGUAGE_NAMES[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    # Save language-specific predictions
    if output_folder:
        lang_output = {
            "predictions": predictions,
            "references": [ref[0] for ref in references]
        }
        with open(os.path.join(output_folder, f"predictions_{lang_code}.json"), "w", encoding="utf-8") as f:
            json.dump(lang_output, f, ensure_ascii=False, indent=2)

    return results, predictions, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                        required=False, help='Model ID from HuggingFace')
    parser.add_argument('--dataset', type=str, default='tharindu/MUNIChus',
                        required=False, help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=None,
                        required=False, help='Number of samples per language (None for all)')
    parser.add_argument('--languages', type=str, nargs='+', default=None,
                        required=False, help='Languages to evaluate (default: all)')

    args = parser.parse_args()

    model_id = args.model_id
    dataset_name = args.dataset
    num_samples = args.num_samples
    eval_languages = args.languages if args.languages else LANGUAGES

    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name}")
    print(f"Languages: {eval_languages}")
    if num_samples:
        print(f"Samples per language: {num_samples}")

    # Create output folder
    output_folder = os.path.join("outputs", "image_captioning", model_id.split('/')[-1])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check available devices
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        num_gpus = torch.xpu.device_count()
        print(f"Number of XPU devices available: {num_gpus}")

    # Clear any existing memory
    clear_memory()

    # Load model and processor
    print(f"Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "50GiB", 1: "50GiB", 2: "50GiB"},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # Check device distribution
    if hasattr(model, 'hf_device_map'):
        print("\nModel device map:")
        device_distribution = {}
        for name, device in model.hf_device_map.items():
            device_distribution[device] = device_distribution.get(device, 0) + 1
        for device, count in sorted(device_distribution.items()):
            print(f"  {device}: {count} modules")

    print("Model loaded successfully!")
    log_gpu_memory()

    # Evaluate all languages
    all_results = []
    all_predictions = {}
    all_references = {}

    for lang in eval_languages:
        try:
            results, preds, refs = evaluate_language(
                model,
                processor,
                lang,
                dataset_name=dataset_name,
                num_samples=num_samples,
                output_folder=output_folder
            )
            all_results.append(results)
            all_predictions[lang] = preds
            all_references[lang] = [ref[0] for ref in refs]

            log_gpu_memory()

        except Exception as e:
            print(f"Error evaluating {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS SUMMARY - {model_id}")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Save results
    results_file = os.path.join(output_folder, "evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")

    # Save detailed predictions
    predictions_file = os.path.join(output_folder, "all_predictions.json")
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump({
            "predictions": all_predictions,
            "references": all_references
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Predictions saved to {predictions_file}")

    # Calculate and save summary statistics
    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Image Captioning Evaluation Results\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Languages: {eval_languages}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Language Results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write("Average Scores Across All Languages:\n")
        f.write(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}\n")
        f.write(f"  Average chrF:   {results_df['chrf'].mean():.2f}\n")
        f.write(f"  Average CIDEr:  {results_df['cider'].mean():.2f}\n")

        # CJK-specific averages
        cjk_results = results_df[results_df['language'].isin(CJK_LANGUAGES)]
        if not cjk_results.empty:
            f.write("\nAverage Scores for CJK Languages (ja, zh, yue):\n")
            f.write(f"  Average BLEU-4: {cjk_results['bleu4'].mean():.2f}\n")
            f.write(f"  Average chrF:   {cjk_results['chrf'].mean():.2f}\n")
            f.write(f"  Average CIDEr:  {cjk_results['cider'].mean():.2f}\n")

        # Non-CJK averages
        non_cjk_results = results_df[~results_df['language'].isin(CJK_LANGUAGES)]
        if not non_cjk_results.empty:
            f.write("\nAverage Scores for Non-CJK Languages:\n")
            f.write(f"  Average BLEU-4: {non_cjk_results['bleu4'].mean():.2f}\n")
            f.write(f"  Average chrF:   {non_cjk_results['chrf'].mean():.2f}\n")
            f.write(f"  Average CIDEr:  {non_cjk_results['cider'].mean():.2f}\n")

        # GPU info
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            f.write(f"\nGPU Configuration:\n")
            f.write(f"  Number of GPUs: {torch.xpu.device_count()}\n")
            for i in range(torch.xpu.device_count()):
                allocated = torch.xpu.memory_allocated(i) / 1024 ** 3
                reserved = torch.xpu.memory_reserved(i) / 1024 ** 3
                f.write(f"  GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB\n")

    print(f"✓ Summary saved to {summary_file}")

    # Print averages
    print("\nAverage Scores Across All Languages:")
    print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")

    log_gpu_memory()


if __name__ == '__main__':
    main()