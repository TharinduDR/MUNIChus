from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class MUNIChusLoadConfig:
    dataset_id: str = "tharindu/MUNIChus"
    languages: Tuple[str, ...] = ("ar", "en", "fr", "hi", "id", "ja", "si", "ur", "yue", "zh")
    cjk_tokens: Tuple[str,...] = ("ja", "zh", "yue")
    language_names: Dict[str, str] = field(default_factory=lambda: {
        "ar": "Arabic",
        "en": "English",
        "fr": "French",
        "hi": "Hindi",
        "id": "Indonesian",
        "ja": "Japanese",
        "si": "Sinhala",
        "ur": "Urdu",
        "yue": "Cantonese",
        "zh": "Chinese",
    })
    split: str = "train"
    batch_size: int = 1
    num_workers: int = 4
    remove_empty: bool = True
    # Max rows per-language for quick smoke tests, None for full
    max_rows_per_lang: Optional[int] = None
    seed: int = 42




@dataclass
class BasicParams:
    seed:int = 42

    project_root_dir:str = "/projects/mzampier/tsuyog/MUNIChus/inft"
    logs_dir: str = f"{project_root_dir}/logs"
    output_dir: str = f"{project_root_dir}/outputs"
    metric_dir: str = f"{output_dir}/metrics"

    llama_model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llama_processor_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    aya_model_name: str = "CohereLabs/aya-vision-8b"
    aya_processor_name: str = "CohereLabs/aya-vision-8b"

    llama_hf_gen_file: str = f"{output_dir}/hf_gen/llama32_hf.jsonl"
    aya_hf_gen_file: str = f"{output_dir}/hf_gen/aya_hf.jsonl"

    llama_inft_gen_file: str = f"{output_dir}/inft_gen/llama32_inft.jsonl"
    aya_inft_gen_file: str = f"{output_dir}/inft_gen/aya_inft.jsonl"

    attn_impl: str = "eager"  # "eager" | "flash" | "triton"
    dtype: str = "bfloat16"

    # inference
    gen_do_sample: bool = True
    gen_top_p: float = 0.9
    gen_temperature: float = 0.7
    gen_num_beams: int = 1
    gen_max_new_tokens: int = 96
    prefer_sft: bool = True



    # knobs for testing
    run_forward: bool = False           # set True to smoke-test a forward pass
    device_map: Optional[str] = None    # "auto"
                 
    max_length: int = 4096 #4096 #seen in example 



@dataclass
class QLoRAParams:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16" 


@dataclass
class LoRAParams:
    r: int = 64
    alpha: int = 32
    dropout: float = 0.10
    bias: str = "none"
    # attention + MLP projections
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",)
    advanced_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","mm_projector", "multi_modal_projector", "vision_proj")
    aya_advanced_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","multi_modal_projector.linear_1", "multi_modal_projector.linear_2",)





@dataclass
class SFTParams:
    num_epochs: int = 1
    num_workers: int = 4
    lr: float = 1.5e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 100
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    gradient_accumulation_steps: int = 16
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    llama_32_output_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/llama_32_sft"
    llama32_best_model_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/llama_32_sft/best_model"
    # advaced llama
    llama32_adv_output_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/llama32_adv_sft"
    llama32_adv_best_model_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/llama32_adv_sft/best_model"

    aya_output_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/aya_sft"
    aya_best_model_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/aya_sft/best_model"
    # advanced aya
    aya_adv_output_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/aya_adv_sft"
    aya_adv_best_model_dir: str = f"/projects/mzampier/tsuyog/MUNIChus/inft/outputs/aya_adv_sft/best_model"

    inference_batch_size: int = 1
    fp16: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    seed: int = 42 


PROMPT_TMPL = (
    "You are writing a caption for a newspaper image.\n"
    "Given the image and this news article excerpt:\n"
    "{news}\n"
    "Task: Write a concise, informative caption for this image in {language}.\n"
    "Guidelines:\n"
    "- Write in {language} language only\n"
    "- Keep it brief\n"
    "- Identify and include: people's names, locations, and organisations\n"
    "- Connect what you see in the image to the news context\n"
    "- Use journalistic style (factual, clear, objective)\n"
    "- Focus on the main subject of the image\n"
    "Caption in {language}:"
)