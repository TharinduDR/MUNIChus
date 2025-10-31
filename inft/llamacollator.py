from typing import List, Dict
import torch
from transformers import AutoProcessor


class LlamaCollator:
    """
    Collator for meta-llama/Llama-3.2-11B-Vision-Instruct.

    Modes
    -----
    - Training (add_generation_prompt=False):
        messages = [
          {"role":"user", "content":[{"type":"image"}, {"type":"text","text": prompt}]},
          {"role":"assistant", "content":[{"type":"text","text": target}]}
        ]
        Labels: mask user-turn (prompt) + padding with -100 → loss only on caption.

    - Inference (add_generation_prompt=True):
        messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text","text": prompt}]}]
        Labels: all -100 (no loss).

    Expected batch item keys
    ------------------------
    - "image": PIL.Image.Image
    - "input_text": str (user prompt WITHOUT any image token)
    - "target_text": str (optional; ignored in inference mode)
    """

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int = 4096,
        input_data_format: str = "channels_last",
        add_generation_prompt: bool = False,
    ):
        if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
            raise ValueError("AutoProcessor must include a tokenizer for Llama 3.2 Vision.")
        self.processor = processor
        self.tok = processor.tokenizer
        self.input_data_format = input_data_format
        self.add_generation_prompt = add_generation_prompt

        # tokenizer/model cap
        tok_cap = getattr(self.tok, "model_max_length", max_length)
        self.eff_max_len = int(min(max_length, tok_cap))

    
    @staticmethod
    def _user_only_message(prompt: str):
        # Image placeholder is handled by the processor; the paired PIL image is passed separately.
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

    @staticmethod
    def _user_and_assistant_messages(prompt: str, target: str):
        return [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]

    def _apply_chat_and_tokenize(
        self,
        messages_batch: List[List[Dict]],
        images: List,
        add_generation_prompt: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        1) Build chat-formatted strings with apply_chat_template (batch-aware).
        2) Tokenize paired (image, text) with the processor.
        """
        # Chat template → text inputs (batch)
        # apply_chat_template returns a *list of strings* for a list of conversations
        texts = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        images = [[img] for img in images]

        # Tokenize (image + text). Batch call with lists.
        proc = self.processor(
            images,                      # List[PIL.Image]
            texts,                       # List[str]
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.eff_max_len,
            input_data_format=self.input_data_format,
            return_tensors="pt",
        )
        return proc

    
    def _encode_infer(self, images: List, prompts: List[str]) -> Dict[str, torch.Tensor]:
        messages_batch = [self._user_only_message(p) for p in prompts]
        proc = self._apply_chat_and_tokenize(messages_batch, images, add_generation_prompt=True)

        # Inference: labels all -100
        labels = torch.full_like(proc["input_ids"], -100)
        labels[proc["attention_mask"] == 0] = -100

        batch: Dict[str, torch.Tensor] = {
            "input_ids": proc["input_ids"],
            "attention_mask": proc["attention_mask"],
            "labels": labels,
        }
        for k in ("pixel_values", "image_sizes", "pixel_attention_mask", "patch_attention_mask", "aspect_ratio_ids", "aspect_ratio_mask"):
            if k in proc and isinstance(proc[k], torch.Tensor):
                batch[k] = proc[k]
        return batch

    def _encode_train(self, images: List, prompts: List[str], targets: List[str]) -> Dict[str, torch.Tensor]:
        # Full conversation (user + assistant) → to compute input_ids and attention
        messages_full = [self._user_and_assistant_messages(p, t) for p, t in zip(prompts, targets)]
        proc_full = self._apply_chat_and_tokenize(messages_full, images, add_generation_prompt=False)

        # User-only conversations → to compute how many tokens to mask as prompt
        messages_user = [self._user_only_message(p) for p in prompts]
        proc_user = self._apply_chat_and_tokenize(messages_user, images, add_generation_prompt=False)

        input_ids      = proc_full["input_ids"]
        attention_mask = proc_full["attention_mask"]
        labels         = input_ids.clone()

        # Mask the entire user-turn (prompt) and padding → loss only on assistant caption
        user_lengths = (proc_user["attention_mask"] == 1).sum(dim=1)  # (B,)
        for i in range(input_ids.size(0)):
            plen = int(user_lengths[i].item())
            labels[i, :plen] = -100
        labels[attention_mask == 0] = -100

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for k in ("pixel_values", "image_sizes", "pixel_attention_mask", "patch_attention_mask"):
            if k in proc_full and isinstance(proc_full[k], torch.Tensor):
                batch[k] = proc_full[k]
        return batch

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images  = [ex["image"] for ex in batch]
        prompts = [ex["input_text"] for ex in batch]

        if self.add_generation_prompt:
            return self._encode_infer(images, prompts)

        targets = [ex.get("target_text", "").strip() for ex in batch]
        return self._encode_train(images, prompts, targets)
