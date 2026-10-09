import os
import random
import re
import time
import math
import base64
from io import BytesIO
from litellm import completion
import numpy as np
from PIL import Image, ImageDraw


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Raise exceptions for any errors not specified
            except Exception as e:
                if (
                    "rate limit" in str(e).lower()
                    or "timed out" in str(e)
                    or "Too Many Requests" in str(e)
                    or "Forbidden for url" in str(e)
                    or "the maximum usage" in str(e).lower()
                    or "server had an error" in str(e).lower()
                    or "has no attribute 'upper'" in str(e).lower()
                    or "internal" in str(e).lower()
                ):
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    # Sleep for the delay
                    time.sleep(delay)
                else:
                    print(str(e))
                    return None

    return wrapper


def get_message_content(response) -> str:
    """Safely extract content or reasoning_content from a completion response."""
    if not response or not hasattr(response, "choices") or not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if not content:
        content = getattr(message, "reasoning_content", None)
    if not content:
        psf = getattr(message, "provider_specific_fields", {}) or {}
        if isinstance(psf, dict):
            content = psf.get("reasoning_content", None)
    if content is None:
        return ""
    return str(content).strip()

def requires_single_image_per_prompt(config: dict) -> bool:
    model_name = str(config.get("model_name", "")).lower()
    api_base = str(config.get("api_base", "")).lower()
    return (
        "integrate.api.nvidia.com" in api_base
        and "meta/llama-3.2-11b-vision-instruct" in model_name
    )


def encode_frame_contact_sheet(frames, timestamps) -> str:
    images = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame).astype(np.uint8))
        scale = 256 / min(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        images.append(image.resize(size, Image.BICUBIC))

    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    tile_width = images[0].width
    tile_height = images[0].height + 28
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (image, timestamp) in enumerate(zip(images, timestamps)):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        draw.rectangle((x, y, x + tile_width, y + 27), fill="black")
        draw.text((x + 6, y + 6), f"{float(timestamp):.1f}s", fill="white")
        sheet.paste(image, (x, y + 28))

    output = BytesIO()
    sheet.save(output, format="JPEG", quality=85)
    return base64.b64encode(output.getvalue()).decode("utf-8")


@retry_with_exponential_backoff
def call_llm_api(
    model_name: str,
    messages: list,
    api_base: str,
    api_key: str = None,
    api_version: str = None,
    max_tokens: int = 32768,
    reasoning_effort: str = None,
    seed: int = 42,
    temperature: float = 1.0,
    tools: list = None,
    tool_choice: str = None,
    return_json: bool = False,
) -> dict:
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    kwargs = {
        "model": model_name,
        "messages": messages,
        "api_base": api_base,
        "api_key": api_key,
        "api_version": api_version,
        "max_completion_tokens": max_tokens,
        "seed": seed,
        "temperature": temperature,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": {"type": "json_object"} if return_json else None,
        "timeout": 900,
    }
    if (
        tool_choice == "required"
        and api_base
        and "integrate.api.nvidia.com" in api_base.lower()
        and "meta/llama-3.2-11b-vision-instruct" in model_name.lower()
    ):
        kwargs["tool_choice"] = "auto"
    if reasoning_effort is not None:
        if ("kimi" in model_name.lower() or "moonshot" in model_name.lower()) and reasoning_effort == "medium":
            reasoning_effort = "low"
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        return completion(**kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "tools" in kwargs and ("tool" in err_str or "invalid json" in err_str or "python_tag" in err_str or "validation error" in err_str or "extra_fields" in err_str):
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            try:
                return completion(**kwargs)
            except Exception:
                pass
        if "response_format" in kwargs and ("json" in err_str or "expecting property name" in err_str or "double quotes" in err_str or "extra_fields" in err_str):
            kwargs.pop("response_format", None)
            try:
                return completion(**kwargs)
            except Exception:
                pass
        if "reasoning_effort" in kwargs and ("thinking_effort" in err_str or "reasoning_effort" in err_str or "unsupported" in err_str):
            kwargs.pop("reasoning_effort", None)
            return completion(**kwargs)
        raise e



def load_subtitles(subtitle_path: str):
    """Parse SRT file and return list of {start_time, end_time, subtitle} dicts."""
    if subtitle_path is None or not os.path.exists(subtitle_path):
        return []
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = []
    # SRT format: index, HH:MM:SS,mmm --> HH:MM:SS,mmm, then text lines
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    blocks = re.split(r"\n\n+", content.strip())

    def to_seconds(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for block in blocks:
        match = pattern.search(block)
        if match:
            start = to_seconds(*match.groups()[:4])
            end = to_seconds(*match.groups()[4:8])
            text = block[match.end() :].strip().replace("\n", " ")
            result.append(
                {
                    "start_time": round(start, 1),
                    "end_time": round(end, 1),
                    "subtitle": text,
                }
            )
    return result


def convert_to_free_form_text_representation(
    history: list[dict], content_type: str = "caption"
) -> str:
    """
    This function will form the textual representation for the entire video to be used for QA.
    It gives a good structured representation of the entire video.
    JSON types of representations are good for outputs, but free-form/ semi-structured should be better for input.
    """
    free_form_text_representation = ""
    if len(history) == 0:
        return f"No {content_type} found."
    for i in history:
        if i[content_type] is None:
            continue
        x = ""
        start_time, end_time = i["start_time"], i["end_time"]
        x += f"**Timestamp**: {start_time}s - {end_time}s\n"
        x += f"**{content_type.capitalize()}**: {i[content_type]}\n"

        free_form_text_representation += f"{x}\n"
    return free_form_text_representation


def create_contact_sheet(frames: list, timestamps: list, max_cols: int = 2) -> Image.Image:
    """Combines multiple video frames into a single grid contact sheet with timestamp overlays."""
    if not frames:
        return Image.new("RGB", (384, 216), (0, 0, 0))

    num_frames = len(frames)
    cols = min(num_frames, max_cols)
    rows = math.ceil(num_frames / cols)

    target_w, target_h = 384, 216
    grid_w = cols * target_w
    grid_h = rows * target_h

    contact_sheet = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    draw = ImageDraw.Draw(contact_sheet)

    for i, (frame, t_str) in enumerate(zip(frames, timestamps)):
        r = i // cols
        c = i % cols
        x_off = c * target_w
        y_off = r * target_h

        img = Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
        contact_sheet.paste(img, (x_off, y_off))

        label_text = f" t={t_str} "
        draw.rectangle([x_off + 4, y_off + 4, x_off + 110, y_off + 24], fill=(0, 0, 0))
        draw.text((x_off + 8, y_off + 6), label_text, fill=(255, 255, 255))

    return contact_sheet