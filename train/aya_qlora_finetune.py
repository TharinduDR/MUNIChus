import torch
from transformers import (
    AutoProcessor,
    AutoModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    pipeline
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import json
import base64
from io import BytesIO
from PIL import Image
import os
import wandb
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import argparse

# For CJK tokenization during evaluation
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: CJK tokenization not available")
    CJK_TOKENIZATION_AVAILABLE = False

# Metrics for evaluation
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config we are going to fine-tune."""
    model_name_or_path: str = field(
        default="CohereLabs/aya-vision-8b",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Whether to use auth token for accessing the model"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input."""
    dataset_name: str = field(
        default="tharindu/MUNIChus",
        metadata={"help": "The name of the dataset to use"}
    )
    languages: str = field(
        default="ar,en,fr,hi,id,ja,si,ur,yue,zh",
        metadata={"help": "Comma-separated list of language codes to use"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging, truncate the number of training examples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging, truncate the number of evaluation examples"}
    )
    val_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of training data to use for validation"}
    )
    max_news_length: int = field(
        default=1200,
        metadata={"help": "Maximum length of news article to include in prompt"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for preprocessing"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank parameter"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout parameter"}
    )
    target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use nested quantization"}
    )


class AyaVisionDataset(torch.utils.data.Dataset):
    """Custom dataset for Aya Vision fine-tuning"""

    def __init__(self, dataset, processor, language_names, max_news_length=1200):
        self.dataset = dataset
        self.processor = processor
        self.language_names = language_names
        self.max_news_length = max_news_length

    def __len__(self):
        return len(self.dataset)

    def pil_image_to_url(self, image):
        """Convert PIL Image to data URL for Aya Vision"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_base64}"

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get language info
        lang_code = item.get('language', 'en')
        language = self.language_names.get(lang_code, 'English')

        # Create the journalistic prompt (same as evaluation)
        prompt = f"""You are writing a caption for a newspaper image.

Given the image and this news article excerpt:
{item['content'][:self.max_news_length]}

Task: Write a concise, informative caption for this image in {language}.

Guidelines:
- Write in {language} language only
- Keep it brief (10-12 words)
- Identify and include: people's names, locations, and organizations visible in the image
- Connect what you see in the image to the news context
- Use journalistic style (factual, clear, objective)
- Focus on the main subject of the image

Caption in {language}:"""

        # Convert image to URL format
        image_url = self.pil_image_to_url(item['image'])

        # Format message with the aya-vision chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Add the target caption as assistant response
        full_text = f"{prompt}\n{item['caption']}"

        # Process with the processor
        try:
            # For Aya Vision, we need to handle the special format
            inputs = self.processor(
                text=[full_text],  # Pass as list
                images=[item['image']],  # Pass image as list
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2048
            )
        except:
            # Fallback approach if the above doesn't work
            inputs = self.processor(
                text=full_text,
                images=item['image'],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2048
            )

        # Flatten batch dimension
        inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}

        # Create labels
        if 'input_ids' in inputs:
            inputs['labels'] = inputs['input_ids'].clone()
            # Set padding tokens to -100 in labels
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'pad_token_id'):
                pad_token_id = self.processor.tokenizer.pad_token_id
            else:
                pad_token_id = 0  # Default fallback
            inputs['labels'][inputs['labels'] == pad_token_id] = -100

        return inputs


def setup_model_and_tokenizer(model_args, lora_args):
    """Setup model with QLoRA configuration"""

    # QLoRA configuration
    compute_dtype = getattr(torch, lora_args.bnb_4bit_compute_dtype)

    if lora_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=lora_args.use_nested_quant,
            bnb_4bit_quant_type=lora_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load model - use AutoModel with trust_remote_code for Aya Vision
    print(f"Loading model: {model_args.model_name_or_path}")
    try:
        # First try loading with AutoModel and trust_remote_code
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.use_auth_token,
            trust_remote_code=True,
            torch_dtype=compute_dtype
        )
    except Exception as e:
        print(f"Failed to load with AutoModel, trying pipeline approach: {e}")
        # Alternative: Load via pipeline and extract model
        pipe = pipeline(
            model=model_args.model_name_or_path,
            task="image-text-to-text",
            device_map="auto",
            model_kwargs={
                "quantization_config": bnb_config,
                "torch_dtype": compute_dtype,
                "trust_remote_code": True
            }
        )
        model = pipe.model

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration - adjust target modules for Aya Vision architecture
    # Common modules in vision-language models
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2",  # For vision encoder
        "lm_head"  # Language model head
    ]

    # Find actual linear modules in the model
    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Extract the actual module name (last part after .)
            module_name = name.split('.')[-1]
            linear_modules.add(module_name)

    print(f"Found linear modules: {linear_modules}")

    # Use intersection of target modules and actual modules
    target_modules = list(set(target_modules) & linear_modules)
    if not target_modules:
        # Fallback to all linear modules if no intersection
        target_modules = list(linear_modules)

    print(f"Using target modules for LoRA: {target_modules}")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=["lm_head"],  # Save the language model head
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    print("Model configuration:")
    model.print_trainable_parameters()

    return model, processor


def load_and_prepare_dataset(data_args, processor, language_names):
    """Load and prepare the MUNIChus dataset with train/val split from training set"""

    languages = data_args.languages.split(',')

    all_train = []
    all_val = []

    for lang in languages:
        print(f"Loading dataset for {lang}...")
        dataset = load_dataset(data_args.dataset_name, lang)

        # Add language code to each sample
        def add_language(example):
            example['language'] = lang
            return example

        # Use only the training set and split it 90/10 for train/val
        train_data = dataset['train'].map(add_language)

        # Limit samples if specified (before splitting)
        if data_args.max_train_samples:
            train_data = train_data.select(range(min(len(train_data), data_args.max_train_samples)))

        # Split train data into 90% train, 10% validation
        train_data = train_data.train_test_split(
            test_size=0.1,
            seed=42,
            shuffle=True
        )

        train_split = train_data['train']
        val_split = train_data['test']  # This is actually validation from train set

        # Apply eval limit if specified
        if data_args.max_eval_samples:
            val_split = val_split.select(range(min(len(val_split), data_args.max_eval_samples)))

        all_train.append(train_split)
        all_val.append(val_split)

        print(f"  {lang}: {len(train_split)} train, {len(val_split)} val samples")

    # Concatenate all languages
    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets(all_train)
    eval_dataset = concatenate_datasets(all_val)

    # Shuffle the concatenated datasets
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)

    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(eval_dataset)}")
    print("Note: Test set is reserved for final evaluation and not used during training")

    # Create custom datasets
    train_dataset = AyaVisionDataset(
        train_dataset,
        processor,
        language_names,
        data_args.max_news_length
    )
    eval_dataset = AyaVisionDataset(
        eval_dataset,
        processor,
        language_names,
        data_args.max_news_length
    )

    return train_dataset, eval_dataset


class AyaVisionTrainer(Trainer):
    """Custom trainer for Aya Vision with evaluation metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chrf_metric = CHRF()
        self.cider_scorer = Cider()
        self.language_names = {
            "ar": "Arabic", "en": "English", "fr": "French",
            "hi": "Hindi", "id": "Indonesian", "ja": "Japanese",
            "si": "Sinhala", "ur": "Urdu", "yue": "Cantonese", "zh": "Chinese"
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for vision-language model"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Shift labels for autoregressive loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()

    # Add argument groups
    parser.add_argument("--output_dir", type=str, default="./aya-vision-qlora-finetuned")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="aya-vision-qlora", name=args.output_dir)

    # Setup arguments
    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    lora_args = LoRAArguments()

    # Language names mapping
    language_names = {
        "ar": "Arabic", "en": "English", "fr": "French",
        "hi": "Hindi", "id": "Indonesian", "ja": "Japanese",
        "si": "Sinhala", "ur": "Urdu", "yue": "Cantonese", "zh": "Chinese"
    }

    # Setup model and processor
    model, processor = setup_model_and_tokenizer(model_args, lora_args)

    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        data_args, processor, language_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=3,
        remove_unused_columns=False,
        fp16=False,  # Use bf16 instead
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb" if args.use_wandb else "none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        group_by_length=True,
    )

    # Create trainer
    trainer = AyaVisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=lambda x: {
            k: torch.stack([item[k] for item in x])
            for k in x[0].keys()
        }
    )

    # Start training
    print("Starting training...")
    print("Note: Validation is done on 10% of training data")
    print("Test set is reserved for final evaluation after training")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save LoRA weights separately
    model.save_pretrained(f"{args.output_dir}/lora_weights")

    print(f"Training complete! Model saved to {args.output_dir}")

    # Push to hub if requested
    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing model to hub: {args.hub_model_id}")
        model.push_to_hub(args.hub_model_id)
        processor.push_to_hub(args.hub_model_id)


if __name__ == "__main__":
    main()