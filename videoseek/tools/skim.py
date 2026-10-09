import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videoseek.utils import (
    call_llm_api,
    convert_to_free_form_text_representation,
    encode_frame_contact_sheet,
    get_message_content,
    requires_single_image_per_prompt,
)
from config import general_config


skim_tool = {
    "type": "function",
    "function": {
        "name": "skim",
        "description": "To localize moments related to the query, quickly scan of a long segment (> {skim_num_frames}s) by sampling {skim_num_frames} frames from the video segment (start_time - end_time).".format(skim_num_frames=general_config["frame_sampling_factor"] * general_config["skim_base"]),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to skim the video. The query should be a concise question that can be answered by the video.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video to skim.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video to skim.",
                },
            },
            "required": ["query", "start_time", "end_time"],
            "additionalProperties": False,
        },
    },
}


def execute_skim(config: dict, parameters: dict) -> str:
    """
    Execute the skim tool using a single contact sheet image grid.
    """
    vr = parameters['vr']
    subtitles = parameters.get('subtitles', [])
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type='subtitle')

    query = parameters.get('query', 'Describe the video segment.')
    start_time = float(parameters.get('start_time', 0))
    end_time = float(parameters.get('end_time', len(vr) / vr.get_avg_fps()))

    fps = vr.get_avg_fps()
    start_idx = max(0, int(start_time * fps))
    end_idx = min(len(vr) - 1, int(end_time * fps))

    num_frames = config.get("frame_sampling_factor", 1) * config.get("skim_base", 2)
    indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=int)

    raw_frames = [vr[idx].asnumpy() for idx in indices]
    timestamps = [f"{idx / fps:.1f}s" for idx in indices]

    cs_img = create_contact_sheet(raw_frames, timestamps, max_cols=2)
    buf = BytesIO()
    cs_img.save(buf, format="jpeg")
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": (
                f"Video segment ({start_time:.1f}s - {end_time:.1f}s):\n"
                f"Frame Timestamps in Grid: {', '.join(timestamps)}\n"
                f"Video Subtitles:\n{subtitles_str}\n\n"
                f"Question / Query:\n{query}\n\n"
                "Please describe the content of the viewed video frames in detail with their timestamps (~25 words per timestamp)."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
    ]

    response = call_llm_api(
        messages=[{"role": "user", "content": content}],
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        api_version=config.get("api_version"),
        max_tokens=config.get("max_tokens", 4096),
        reasoning_effort=config.get("reasoning_effort"),
        seed=config.get("seed", 42),
        temperature=config.get("temperature", 0.2),
    )

    res_str = get_message_content(response)
    if not res_str or set(res_str) == {"!"}:
        return "Failed to extract visual observations. Ensure the model supports multi-image vision input."
    return res_str
