import json
import base64
import re
from io import BytesIO
import numpy as np
from PIL import Image

from videoseek.utils import (
    call_llm_api,
    get_message_content,
    convert_to_free_form_text_representation,
    create_contact_sheet,
)
from config import general_config


overview_tool = {
    "type": "function",
    "function": {
        "name": "overview",
        "description": "To get a structured video summary for the entire video.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
}


def execute_overview(config: dict, parameters: dict) -> str:
    """
    Execute the overview tool using a single contact sheet image grid.
    """
    vr = parameters['vr']
    duration = round(len(vr) / vr.get_avg_fps(), 1)
    subtitles = parameters['subtitles']
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type='subtitle')

    num_frames = config.get("frame_sampling_factor", 1) * config.get("overview_base", 4)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    timestamps = [f"{idx / fps:.1f}s" for idx in frame_indices]
    raw_frames = [vr[idx].asnumpy() for idx in frame_indices]

    cs_img = create_contact_sheet(raw_frames, timestamps, max_cols=2)
    buf = BytesIO()
    cs_img.save(buf, format="jpeg")
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": (
                f"Video Duration: 0.0s - {duration:.1f}s\n"
                f"Video Subtitles:\n{subtitles_str}\n\n"
                f"Frame Timestamps in Grid: {', '.join(timestamps)}\n\n"
                "Please generate detailed descriptions for each frame timestamp shown in the grid.\n"
                "Return ONLY valid JSON. Use this exact schema:\n"
                "{\"frames\": [{\"timestamp\": \"1.0s\", \"description\": \"FRAME_DESCRIPTION_1\"}, ...]}\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
    ]

    response = call_llm_api(
        messages=[{"role": "user", "content": content}],
        model_name=config['model_name'],
        api_base=config['api_base'],
        api_key=config['api_key'],
        api_version=config.get('api_version'),
        max_tokens=config.get('max_tokens', 4096),
        reasoning_effort=config.get('reasoning_effort'),
        seed=config.get('seed', 42),
        temperature=config.get('temperature', 0.2),
        return_json=True,
    )

    raw_content = get_message_content(response)
    if not raw_content or set(raw_content) == {"!"}:
        return "Failed to extract overview descriptions. Ensure the model supports vision inputs."
    raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE)
    raw_content = re.sub(r"\s*```$", "", raw_content)
    try:
        frames_data = json.loads(raw_content).get('frames', [])
    except Exception:
        return raw_content
    if not frames_data:
        return raw_content
    return "\n\n".join([f"{frame.get('timestamp', '')}: {frame.get('description', '')}" for frame in frames_data])