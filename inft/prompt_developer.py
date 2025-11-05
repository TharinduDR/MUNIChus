from typing import Dict, Optional
from PIL import Image
from params import PROMPT_TMPL

def build_instruction(example:Dict, lang_display: Optional[str] = None)->str:
    """
    Given one dataset row, render the instruction text using PROMPT_TMPL.
    If lang_display is None, use example['language'] as the display.
    """
    news = example.get("content", "").strip()
    news = str(news)
    news_trim = news[:1200].replace("{", "{{").replace("}", "}}")
    lang = lang_display or example.get("language", "en")
    return PROMPT_TMPL.format(news=news_trim, language=lang)


def to_supervised_record(example:Dict)->Dict:
    """
    Map raw HF row to a trainer-ready record:
      - 'image': PIL.Image
      - 'input_text': instruction (prompt)
      - 'target_text': ground-truth caption
    """
    img = example["image"]
    if not isinstance(img, Image.Image):
        # Just incase if HF delivers images as a dict
        img = Image.open(img).convert("RGB")
    
    return {
        "image": img,
        "input_text": build_instruction(example),
        "target_text": example.get("caption", "").strip(),
        "language": example.get("language", "en"),
        "title": example.get("title", ""),
    }