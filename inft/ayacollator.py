from typing import List, Dict
import torch
from transformers import AutoProcessor


class AyaCollator:
    """
    Collator for CohereLabs/aya-vision-8b using the official chat template.

    Modes
    -----
    - Training (add_generation_prompt=False):
        messages = [
          {"role": "user", "content":[{"type":"image","image": PIL}, {"type":"text","text": prompt}]},
          {"role": "assistant", "content":[{"type":"text","text": target}]}
        ]
        labels: mask all user/prompt tokens + padding with -100 → loss only on target.

    - Inference (add_generation_prompt=True):
        messages = [{"role": "user", "content":[{"type":"image","image": PIL}, {"type":"text","text": prompt}]}]
        labels: all -100 (no loss).

    Expected batch items
    --------------------
    - image: PIL.Image.Image
    - input_text: str  (user prompt WITHOUT any image tag)
    - target_text: str (optional; ignored in inference mode)
    """

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int = 4096,
        input_data_format: str = "channels_last",
        add_generation_prompt: bool = False,
    ):
        if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
            raise ValueError("AutoProcessor must include a tokenizer for Aya Vision.")
        self.processor = processor
        self.tok = processor.tokenizer
        self.input_data_format = input_data_format
        self.add_generation_prompt = add_generation_prompt

        
        tok_cap = getattr(self.tok, "model_max_length", max_length)
        self.eff_max_len = int(min(max_length, tok_cap))

    
    @staticmethod
    def _user_only_message(img, prompt: str):
        # Embed PIL image directly in the message
        return [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text",  "text":  prompt}]}]

    @staticmethod
    def _user_and_assistant_messages(img, prompt: str, target: str):
        return [
            {"role": "user",      "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text":  prompt}]},
            {"role": "assistant", "content": [{"type": "text",  "text":  target}]},
        ]

    
    def _encode_infer(self, images: List, prompts: List[str]) -> Dict[str, torch.Tensor]:
        messages_batch = [
            self._user_only_message(img, p) for img, p in zip(images, prompts)
        ]

        proc = self.processor.apply_chat_template(
            messages_batch,
            padding=True,
            truncation=True,
            max_length=self.eff_max_len,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            input_data_format=self.input_data_format,
        )

        # Labels: all -100 (no loss)
        labels = torch.full_like(proc["input_ids"], -100)
        labels[proc["attention_mask"] == 0] = -100

        return {
            "input_ids": proc["input_ids"],
            "attention_mask": proc["attention_mask"],
            "labels": labels,
            **{k: v for k, v in proc.items() if k in (
                "pixel_values", "image_sizes", "pixel_attention_mask", "patch_attention_mask"
            )},
        }

    def _encode_train(self, images: List, prompts: List[str], targets: List[str]) -> Dict[str, torch.Tensor]:
        # Full pair messages (user + assistant)
        messages_batch = [
            self._user_and_assistant_messages(img, p, t)
            for img, p, t in zip(images, prompts, targets)
        ]

        proc_full = self.processor.apply_chat_template(
            messages_batch,
            padding=True,
            truncation=True,
            max_length=self.eff_max_len,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            input_data_format=self.input_data_format,
        )

        # User-only messages to compute prompt lengths for masking
        user_only_msgs = [
            self._user_only_message(img, p) for img, p in zip(images, prompts)
        ]
        proc_user = self.processor.apply_chat_template(
            user_only_msgs,
            padding=True,
            truncation=True,
            max_length=self.eff_max_len,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            input_data_format=self.input_data_format,
        )

        input_ids      = proc_full["input_ids"]
        attention_mask = proc_full["attention_mask"]
        labels         = input_ids.clone()

        # Mask the user/prompt region + padding → loss only on assistant/caption
        user_lengths = (proc_user["attention_mask"] == 1).sum(dim=1)  # (B,)
        for i in range(input_ids.size(0)):
            plen = int(user_lengths[i].item())
            labels[i, :plen] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            **{k: v for k, v in proc_full.items() if k in (
                "pixel_values", "image_sizes", "pixel_attention_mask", "patch_attention_mask"
            )},
        }

    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images  = [ex["image"] for ex in batch]
        prompts = [ex["input_text"] for ex in batch]

        if self.add_generation_prompt:
            return self._encode_infer(images, prompts)

        targets = [ex.get("target_text", "").strip() for ex in batch]
        return self._encode_train(images, prompts, targets)
