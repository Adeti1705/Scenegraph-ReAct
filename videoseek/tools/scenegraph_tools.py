import json
import base64
import math
import re
import warnings
from io import BytesIO
import numpy as np

from videoseek.utils import (
    call_llm_api,
    get_message_content,
    convert_to_free_form_text_representation,
    create_contact_sheet,
)

def _source_fps(vr) -> float:
    fps = float(vr.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video has invalid source FPS: {fps!r}")
    return fps


def _clip_interval(vr, start_time: float, end_time: float, max_seconds=None):
    frame_count = len(vr)
    if frame_count == 0:
        return 0.0, 0.0
    fps = _source_fps(vr)
    duration = (frame_count - 1) / fps
    start_time = min(max(start_time, 0.0), duration)
    end_time = min(max(end_time, start_time), duration)
    if max_seconds is not None:
        end_time = min(end_time, start_time + max_seconds)
    return start_time, end_time


def _sample_indices(vr, start_time: float, end_time: float, num_frames: int):
    frame_count = len(vr)
    if frame_count == 0:
        return np.array([], dtype=int)
    fps = _source_fps(vr)
    start_time, end_time = _clip_interval(vr, start_time, end_time)
    start_idx = int(start_time * fps)
    end_idx = min(frame_count - 1, int(end_time * fps))
    return np.linspace(start_idx, end_idx, num=max(1, num_frames), dtype=int)


def _ontology_guidance(config: dict) -> str:
    categories = config.get("scenegraph_object_categories", {})
    predicates = config.get("scenegraph_predicates", [])
    return (
        f"PVSG object categories for reference (grouped by is_thing): {json.dumps(categories)}\n"
        f"PVSG predicates for reference: {json.dumps(predicates)}\n"
        "These vocabularies are references, not constraints on raw observations. Use concise, visually grounded open-vocabulary labels; do not force an inexact taxonomy label. Treat subtitles as context, never as visual evidence."
    )

def execute_overview_scenegraph(config: dict, parameters: dict) -> str:
    """Samples frames across video and combines them into a single contact sheet image grid."""
    vr = parameters["vr"]
    subtitles = parameters.get("subtitles", [])
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type="subtitle")

    num_frames = max(1, config.get("frame_sampling_factor", 1) * config.get("overview_base", 4))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError("Cannot run scene-graph overview on a video with no frames")
    fps = _source_fps(vr)
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = [vr[idx].asnumpy() for idx in indices]
    timestamps = [f"{idx / fps:.1f}s" for idx in indices]

    cs_img = create_contact_sheet(frames, timestamps, max_cols=2)
    buf = BytesIO()
    cs_img.save(buf, format="jpeg")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": (
                f"Sampled frame timestamps (seconds): {', '.join(timestamps)}\n"
                f"{_ontology_guidance(config)}\n\n"
                f"Optional subtitle context (do not use it to assert visible objects or relations):\n{subtitles_str}\n\n"
                "Task: Build a comprehensive whole-video object inventory from the displayed frames.\n"
                "1. List EVERY visually distinguishable object, including:\n"
                "   - Foreground actors and props ('thing' categories, e.g., adult, child, baby, food, cake, candle, cup, bag, toy, phone, dog, cat, etc.).\n"
                "   - Environment and structural elements ('stuff' and furniture categories, e.g., wall, floor, countertop, table, chair, cabinet, door, fridge, microwave, window, sofa, etc.).\n"
                "2. Identify distinct multiple instances of the same category (e.g. Adult 1, Adult 2, Child 1, Child 2, Table 1, Cabinet 1, Cake 1) with distinguishing visual attributes (clothing color, position, appearance).\n"
                "3. Describe prominent static spatial and support relations (e.g. object on table, candle on cake, person sitting on chair, cabinet on wall).\n"
                "Do not assign numeric IDs; the finalizer assigns stable IDs."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
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
        return "Failed to extract overview entity inventory. Check vision capability of model."
    return res_str


def execute_skim_scenegraph(config: dict, parameters: dict) -> str:
    """Scans mid-grained video interval into a single contact sheet image grid to detect relationship triplets."""
    vr = parameters["vr"]
    subtitles = parameters.get("subtitles", [])
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type="subtitle")

    fps = _source_fps(vr)
    start_time = float(parameters.get("start_time", 0))
    end_time = float(parameters.get("end_time", len(vr) / fps))
    start_time, end_time = _clip_interval(vr, start_time, end_time)
    query = parameters.get("query", "What interactions and spatial relationships occur in this segment?")

    num_frames = config.get("frame_sampling_factor", 1) * config.get(
        "scenegraph_skim_base", config.get("skim_base", 2)
    )
    indices = _sample_indices(vr, start_time, end_time, num_frames)

    frames = [vr[idx].asnumpy() for idx in indices]
    timestamps = [f"{idx / fps:.1f}s" for idx in indices]

    cs_img = create_contact_sheet(frames, timestamps, max_cols=2)
    buf = BytesIO()
    cs_img.save(buf, format="jpeg")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": (
                f"Segment Interval: {start_time:.1f}s to {end_time:.1f}s\n"
                f"Frame Timestamps in Grid: {', '.join(timestamps)}\n"
                f"{_ontology_guidance(config)}\n"
                f"Query: {query}\n\n"
                f"Optional subtitle context (do not treat as visual evidence):\n{subtitles_str}\n\n"
                "Task: Find ALL visually supported relationships in this interval, including:\n"
                "1. Static spatial, contact, and support relations involving objects, persons, and environment/furniture (e.g., 'on', 'next to', 'sitting on', 'standing on', 'in', 'in front of', 'wearing').\n"
                "2. Dynamic action and interaction relations (e.g., 'holding', 'pointing to', 'picking', 'touching', 'eating', 'opening', 'talking to', 'carrying').\n"
                "3. Any newly appearing objects (both foreground 'things' and background/structural 'stuff') and distinct instances of the same category (e.g., Adult 1, Adult 2, Child 1).\n\n"
                "For each candidate, report:\n"
                "- Subject category and distinguishing attributes (or instance identifier)\n"
                "- Concise predicate phrase (referencing PVSG predicate list)\n"
                "- Object category and distinguishing attributes (or instance identifier)\n"
                "- Temporal bounds in seconds: For ongoing/static relations, span across the timestamps where they co-exist in that state. For transient actions, tightly bound start/end to the specific timestamps where the action occurs. Keep discontinuous intervals separate."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
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
        return "Failed to extract skim relationships. Check vision capability of model."
    return res_str


def execute_focus_scenegraph(config: dict, parameters: dict) -> str:
    """Inspects short clip into a single contact sheet image grid to verify fine-grained predicates."""
    vr = parameters["vr"]
    subtitles = parameters.get("subtitles", [])
    subtitles_str = convert_to_free_form_text_representation(subtitles, content_type="subtitle")
    fps = _source_fps(vr)
    start_time = float(parameters.get("start_time", 0))
    end_time = float(parameters.get("end_time", start_time + 4.0))
    max_focus_seconds = float(config.get("scenegraph_focus_max_seconds", 4.0))
    start_time, end_time = _clip_interval(
        vr, start_time, end_time, max_seconds=max_focus_seconds
    )
    query = parameters.get("query", "Verify precise spatial predicate and contact interaction.")

    num_frames = config.get("frame_sampling_factor", 1) * config.get(
        "scenegraph_focus_base", config.get("focus_base", 2)
    )
    indices = _sample_indices(vr, start_time, end_time, num_frames)

    frames = [vr[idx].asnumpy() for idx in indices]
    timestamps = [f"{idx / fps:.1f}s" for idx in indices]

    cs_img = create_contact_sheet(frames, timestamps, max_cols=2)
    buf = BytesIO()
    cs_img.save(buf, format="jpeg")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": (
                f"Short Clip Interval: {start_time:.1f}s to {end_time:.1f}s\n"
                f"Frame Timestamps in Grid: {', '.join(timestamps)}\n"
                f"{_ontology_guidance(config)}\n"
                f"Query: {query}\n\n"
                f"Optional subtitle context (do not treat as visual evidence):\n{subtitles_str}\n\n"
                "Task: Verify the queried object instances and relationship against the displayed frames.\n"
                "1. Accurately verify static spatial/contact relations (e.g., 'on', 'next to', 'sitting on', 'standing on', 'in', 'in front of', 'wearing') or dynamic action relations ('holding', 'touching', 'pointing to', 'eating', 'picking', etc.).\n"
                "2. Specifically describe the subject and object (including furniture/structural 'stuff' like table, chair, countertop, wall, or specific persons/props).\n"
                "3. Cite the exact sample timestamps in the grid supporting the relation. Provide tight start/end bounds in seconds based on visible evidence (do not guess beyond the observed timestamps). If unverified, report unverified."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
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
        return "Failed to verify focus predicate details. Check vision capability of model."
    return res_str


def _normalize_time_spans(value):
    spans = []

    def parse_time(item):
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            return float(item) if math.isfinite(item) else None
        if isinstance(item, str):
            text = re.sub(r"\s*(seconds?|secs?|s)$", "", item.strip().lower())
            try:
                parsed = float(text)
            except ValueError:
                return None
            return parsed if math.isfinite(parsed) else None
        return None

    def visit(item):
        if isinstance(item, dict):
            start = next((item[key] for key in ("start_seconds", "start_time", "start") if key in item), None)
            end = next((item[key] for key in ("end_seconds", "end_time", "end") if key in item), None)
            start_value = parse_time(start)
            end_value = parse_time(end)
            if start_value is not None and end_value is not None:
                spans.append((start_value, end_value))
                return
            raise ValueError(f"Invalid relation span object: {item!r}")

        if isinstance(item, str):
            match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)?\s*",
                item.lower(),
            )
            if match:
                spans.append((float(match.group(1)), float(match.group(2))))
                return
            raise ValueError(f"Invalid relation span string: {item!r}")

        if isinstance(item, (list, tuple)):
            if len(item) == 2:
                start_value = parse_time(item[0])
                end_value = parse_time(item[1])
                if start_value is not None and end_value is not None:
                    spans.append((start_value, end_value))
                    return
            for nested in item:
                visit(nested)
            return

        raise ValueError(f"Invalid relation span {item!r}; expected two time values")

    visit(value)
    return spans


def _normalize_parsed_dict(data: dict, raw_text: str = "") -> dict:
    if not isinstance(data, dict):
        return data

    if "objects" not in data:
        for k in ("entities", "nodes", "elements", "object_list"):
            if k in data and isinstance(data[k], list):
                data["objects"] = data.pop(k)
                break
        if "objects" not in data:
            data["objects"] = []

    if "relations_seconds" not in data:
        for k in ("relations", "relationships", "relation_seconds", "edges", "triplets"):
            if k in data and isinstance(data[k], list):
                data["relations_seconds"] = data.pop(k)
                break
        if "relations_seconds" not in data:
            data["relations_seconds"] = []

    # If relations_seconds is empty, try regex extracting from raw_text
    if not data["relations_seconds"] and raw_text:
        extracted = []
        for m in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*(\[\[.*?\]\])\s*\]', raw_text, re.DOTALL):
            try:
                spans = json.loads(m.group(4))
                extracted.append([int(m.group(1)), int(m.group(2)), m.group(3).strip(), spans])
            except Exception:
                pass
        if extracted:
            data["relations_seconds"] = extracted

    return data


def repair_and_parse_scenegraph_json(res_str: str) -> dict:
    if not res_str or not isinstance(res_str, str):
        raise ValueError("Empty or invalid scene graph response")

    text = res_str.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _normalize_parsed_dict(data, text)
    except Exception:
        pass

    # 1. Fix missing commas between arrays/objects: ] [ -> ], [ and } { -> }, {
    fixed = re.sub(r"\]\s*\[", "], [", text)
    fixed = re.sub(r"\}\s*\{", "}, {", fixed)
    # 2. Fix trailing commas before closing brackets
    fixed = re.sub(r",\s*([\]\}])", r"\1", fixed)

    try:
        data = json.loads(fixed)
        if isinstance(data, dict):
            return _normalize_parsed_dict(data, text)
    except Exception:
        pass

    # 3. Handle truncation or unbalanced brackets
    for r_idx in range(len(fixed) - 1, 0, -1):
        if fixed[r_idx] in ("]", "}"):
            candidate = fixed[: r_idx + 1]
            open_cur = candidate.count("{") - candidate.count("}")
            open_sq = candidate.count("[") - candidate.count("]")
            balanced = candidate + ("]" * max(0, open_sq)) + ("}" * max(0, open_cur))
            balanced = re.sub(r",\s*([\]\}])", r"\1", balanced)
            try:
                data = json.loads(balanced)
                if isinstance(data, dict) and ("objects" in data or "entities" in data):
                    return _normalize_parsed_dict(data, text)
            except Exception:
                continue

    # 4. Regex fallback extraction if structure is badly damaged
    objects = []
    seen_obj_ids = set()
    for m in re.finditer(r'\{\s*"object_id"\s*:\s*(\d+)\s*,\s*"category"\s*:\s*"([^"]+)"\s*\}', text):
        obj_id = int(m.group(1))
        if obj_id not in seen_obj_ids:
            seen_obj_ids.add(obj_id)
            objects.append({"object_id": obj_id, "category": m.group(2).strip()})

    relations = []
    for m in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*(\[\[.*?\]\])\s*\]', text, re.DOTALL):
        try:
            spans = json.loads(m.group(4))
            relations.append([int(m.group(1)), int(m.group(2)), m.group(3).strip(), spans])
        except Exception:
            pass

    if objects:
        return {"objects": objects, "relations_seconds": relations}

    raise ValueError(f"Could not parse valid scene graph JSON from model response: {res_str[:500]}")


def _to_open_vocabulary_graph(graph, video_id, meta):
    if not isinstance(graph, dict):
        raise ValueError("Scene-graph answer must be a JSON object")

    raw_objects = graph.get("objects") or graph.get("entities") or []
    if not isinstance(raw_objects, list):
        raw_objects = []

    objects = []
    object_ids = set()
    category_to_id = {}
    for idx, item in enumerate(raw_objects, start=1):
        if isinstance(item, dict):
            object_id = item.get("object_id", item.get("id", idx))
            category = str(item.get("category", item.get("label", item.get("name", "")))).strip()
        elif isinstance(item, str):
            object_id = idx
            category = item.strip()
        else:
            continue

        if not isinstance(object_id, int) or object_id in object_ids:
            object_id = max(object_ids, default=0) + 1
        if not category:
            category = "object"
        object_ids.add(object_id)
        objects.append({"object_id": object_id, "category": category})
        category_to_id[category.lower()] = object_id

    raw_relations = (
        graph.get("relations_seconds")
        or graph.get("relations")
        or graph.get("relationships")
        or graph.get("relation_seconds")
        or graph.get("triplets")
        or []
    )
    if not isinstance(raw_relations, list):
        raw_relations = []

    merged_relations = {}
    duration = float(meta["duration"])
    for relation in raw_relations:
        if isinstance(relation, dict):
            subject_id = relation.get("subject_id", relation.get("subject"))
            object_id = relation.get("object_id", relation.get("object"))
            predicate = relation.get("predicate", "")
            spans_seconds = relation.get("spans", relation.get("duration", relation.get("segments", [])))
        elif isinstance(relation, (list, tuple)) and len(relation) >= 4:
            subject_id, object_id, predicate, spans_seconds = relation[:4]
        else:
            continue

        # Resolve string subjects/objects to IDs if needed
        if isinstance(subject_id, str) and subject_id.lower() in category_to_id:
            subject_id = category_to_id[subject_id.lower()]
        if isinstance(object_id, str) and object_id.lower() in category_to_id:
            object_id = category_to_id[object_id.lower()]

        if subject_id not in object_ids or object_id not in object_ids:
            continue
        predicate = str(predicate).strip()
        if not predicate:
            continue
        frame_spans = []
        for start_seconds, end_seconds in _normalize_time_spans(spans_seconds):
            if start_seconds > end_seconds:
                continue
            start_seconds = min(max(start_seconds, 0.0), duration)
            end_seconds = min(max(end_seconds, start_seconds), duration)
            frame_spans.append([start_seconds, end_seconds])

        if not frame_spans:
            continue

        edge_key = (subject_id, object_id, predicate)
        if edge_key not in merged_relations:
            merged_relations[edge_key] = []
        merged_relations[edge_key].extend(frame_spans)

    relations = [
        [sub_id, obj_id, pred, spans]
        for (sub_id, obj_id, pred), spans in merged_relations.items()
    ]

    return {
        "video_id": video_id,
        "meta": meta,
        "objects": objects,
        "relations_seconds": relations,
    }


def to_pvsg_graph(graph, object_categories, predicates):
    """Create a taxonomy-constrained PVSG export without changing the raw graph."""
    categories = object_categories.get("thing", []) + object_categories.get("stuff", [])
    category_set = set(categories)
    thing_categories = set(object_categories.get("thing", []))
    allowed_predicates = set(predicates)
    canonical_objects = []
    object_ids = set()

    for item in graph.get("objects", []):
        object_id = item["object_id"]
        if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id in object_ids:
            raise ValueError("Object IDs must be unique integers")
        object_ids.add(object_id)
        category = item["category"].strip().lower()
        if category not in category_set:
            if "others" not in category_set:
                warnings.warn(f"Skipping non-PVSG object category {category!r}", stacklevel=2)
                continue
            warnings.warn(f"Mapping non-PVSG category {category!r} to 'others' in export", stacklevel=2)
            category = "others"
        canonical_objects.append({
            "object_id": object_id,
            "category": category,
            "is_thing": category in thing_categories,
            "status": [],
        })

    canonical_ids = {item["object_id"] for item in canonical_objects}
    fps = float(graph["meta"]["fps"])
    num_frames = int(graph["meta"]["num_frames"])
    if not math.isfinite(fps) or fps <= 0 or num_frames <= 0:
        raise ValueError("PVSG metadata must have finite positive fps and num_frames")
    max_frame = num_frames - 1
    duration = float(graph["meta"]["duration"])
    canonical_relations = []
    for subject_id, object_id, predicate, spans_seconds in graph.get("relations_seconds", []):
        predicate = predicate.strip().lower()
        if predicate not in allowed_predicates:
            warnings.warn(f"Skipping non-PVSG predicate {predicate!r} in export", stacklevel=2)
            continue
        if subject_id not in canonical_ids or object_id not in canonical_ids:
            continue
        frame_spans = []
        for start_seconds, end_seconds in _normalize_time_spans(spans_seconds):
            start_seconds = min(max(start_seconds, 0.0), duration)
            end_seconds = min(max(end_seconds, start_seconds), duration)
            start_frame = min(max_frame, int(start_seconds * fps + 0.5))
            end_frame = min(max_frame, int(end_seconds * fps + 0.5))
            frame_spans.append([start_frame, max(start_frame, end_frame)])
        canonical_relations.append([subject_id, object_id, predicate, frame_spans])

    return {
        "video_id": graph["video_id"],
        "meta": graph["meta"],
        "objects": canonical_objects,
        "relations": canonical_relations,
    }


def execute_answer_scenegraph(config: dict, parameters: dict) -> str:
    """Consolidate observations into a PVSG-format scene graph."""
    messages = parameters.get("messages", [])
    video_id = parameters.get("video_id", "")
    meta = parameters.get("meta", {})
    object_categories = parameters.get("object_categories", {})
    predicates = parameters.get("predicates", [])
    observations = [
        msg["content"]
        for msg in messages
        if isinstance(msg, dict)
        and msg.get("role") == "tool"
        and isinstance(msg.get("content"), str)
    ]
    observations_summary = "\n\n".join(observations) or "No tool observations were recorded."

    clean_prompt = [
        {
            "role": "system",
            "content": "You format PVSG spatio-temporal scene graphs. Use only the supplied visual observations; do not add unsupported objects or relationships."
        },
        {
            "role": "user",
            "content": (
                f"Video ID: {video_id}\nVideo metadata: {json.dumps(meta)}\n"
                f"PVSG object categories for reference only: {json.dumps(object_categories)}\n"
                f"PVSG predicates for reference only: {json.dumps(predicates)}\n\n"
                f"Observed Video Evidence and Triplets:\n{observations_summary}\n\n"
                "Construct one comprehensive scene graph. Preserve stable object identities and represent discontinuous intervals as separate spans.\n"
                "1. Comprehensive Objects Inventory:\n"
                "   - Include ALL distinct object instances identified in the visual observations.\n"
                "   - Include foreground objects/actors ('thing' categories: all distinct adults, children, props, food, etc.). Assign distinct unique integer `object_id`s to distinct individuals/instances.\n"
                "   - Include background, structural, and furniture elements ('stuff' categories: wall, floor, countertop, table, chair, cabinet, door, fridge, microwave, etc.).\n\n"
                "2. Meaningful Relationships (DO NOT output trivial ambient proximity like 'person next to wall', 'person on floor', 'person next to microwave'):\n"
                "   - Active physical contact & manipulation: `holding`, `picking`, `touching`, `eating`, `carrying`.\n"
                "   - Visual attention & gaze: `looking at`, `pointing to`.\n"
                "   - Direct support & placement: `on`, `in`, `sitting on`, `standing on` (e.g. candle on cake, cake on table, person sitting on chair).\n\n"
                "3. Edge Consolidation & Accurate Temporal Bounds:\n"
                "   - Group ALL intervals for the same (subject, object, predicate) into a SINGLE entry with an array of spans `[[start1, end1], ...]`. Do not output duplicate triplet entries.\n"
                "   - For continuous ongoing relationships observed across multiple chunks (e.g. adult holding child across consecutive intervals), merge them into a unified continuous span `[start_sec, end_sec]`.\n"
                "   - For transient actions (e.g. pointing, picking, blowing), tightly bound start/end to the specific observed timestamps where the action occurs.\n\n"
                "Return one JSON object with `objects`, an array of `{object_id: unique integer, category: open-vocabulary string}`, and `relations_seconds`, an array of `[subject_object_id, object_object_id, predicate_string, [[start_seconds, end_seconds], ...]]` entries.\n"
                "Ensure IDs are unique integers, every relation references valid object IDs, and intervals are ordered numeric seconds within video duration.\n"
                "Return ONLY the valid JSON object without markdown formatting or introductory text."
            )
        }
    ]

    response = call_llm_api(
        messages=clean_prompt,
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        api_version=config.get("api_version"),
        max_tokens=max(8192, config.get("max_tokens", 4096)),
        reasoning_effort=config.get("reasoning_effort"),
        seed=config.get("seed", 42),
        temperature=0.1,
        return_json=True,
    )

    res_str = get_message_content(response)
    graph = repair_and_parse_scenegraph_json(res_str)
    try:
        raw_graph = _to_open_vocabulary_graph(graph, video_id, meta)
    except (TypeError, ValueError) as exc:
        relation_excerpt = repr(graph.get("relations_seconds"))[:1200]
        raise ValueError(f"{exc}; relations_seconds excerpt: {relation_excerpt}") from exc
    return json.dumps(raw_graph, ensure_ascii=False)