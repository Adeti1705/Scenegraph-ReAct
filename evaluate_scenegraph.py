import argparse
from collections import Counter
import json
import os
import re
import time
from pathlib import Path

from config import general_config
from videoseek.scenegraph_agent import SceneGraphAgent
from videoseek.tools.scenegraph_tools import to_pvsg_graph, repair_and_parse_scenegraph_json
from videoseek.utils import call_llm_api, get_message_content


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Spatio-Temporal Scene Graph Generation with VideoSeek")
    p.add_argument("--video_path", default=None, help="Local path or URL to video (for single video evaluation).")
    p.add_argument("--video_dir", default=None, help="Directory containing video files for batch evaluation.")
    p.add_argument("--pred_dir", default=None, help="Eval-only mode: directory of run subdirs containing scenegraph_raw.json; scores saved predictions without running the agent.")
    p.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate when using --video_dir (default: 10).")
    p.add_argument("--gt_file", default=None, help="Optional ground truth scene graph JSON file for metric calculation.")
    p.add_argument("--subtitle_path", default=None, help="Optional subtitle file path.")
    p.add_argument("--output_dir", default="./output/scenegraph_eval", help="Output directory.")
    p.add_argument("--model_name", default="openai/meta/llama-3.2-11b-vision-instruct", help="NVIDIA NIM model name for agent generation.")
    p.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", help="API base URL for agent generation.")
    p.add_argument("--api_key", default=None, help="API key for agent generation.")
    p.add_argument("--judge_model_name", default=None, help="Optional model name for semantic judging/alignment (default: uses --model_name).")
    p.add_argument("--judge_api_base", default=None, help="Optional API base URL for semantic judging (default: uses --api_base).")
    p.add_argument("--judge_api_key", default=None, help="Optional API key for semantic judging (default: uses --api_key or env).")
    p.add_argument("--frame_sampling_factor", type=int, default=1, help="Frame sampling multiplier for NIM API.")
    p.add_argument("--annotation_fps", type=float, default=5.0, help="Frame rate used by benchmark relation spans when no ground-truth metadata is supplied.")
    p.add_argument("--taxonomy_file", default=None, help="PVSG dataset JSON containing the object and predicate vocabularies.")
    p.add_argument("--lenient_semantic", action="store_true", help="Also score with a pairwise lexical semantic aligner using the judge model; incurs extra API calls.")
    p.add_argument("--tiou_thresholds", default="0.1,0.3,0.5", help="Comma-separated tIoU thresholds for evaluation (e.g. 0.1,0.3,0.5).")
    p.add_argument("--split", default=None, choices=["val", "train", "all"], help="Dataset split to evaluate from gt_file (e.g. 'val' or 'train').")
    p.add_argument("--dataset_name", default="vidor", help="Dataset name inside ground-truth split dict (default: 'vidor').")
    p.add_argument("--max_tokens", type=int, default=4096, help="Max tokens.")
    p.add_argument("--max_steps", type=int, default=10, help="Max agent steps.")
    p.add_argument("--verbose", action="store_true", help="Print step logs.")
    return p.parse_args()


def _norm_predicate(predicate) -> str:
    """Canonicalize predicate text: underscores/hyphens become spaces so that
    'in_front_of' matches the PVSG predicate 'in front of'."""
    return re.sub(r"[_\-\s]+", " ", str(predicate)).strip().lower()


def _normalize_graph(graph: dict):
    objects = graph.get("objects", graph.get("entities", []))
    object_labels = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id", item.get("id"))
        label = item.get("category", item.get("label"))
        if object_id is not None and label is not None:
            object_labels[object_id] = str(label).strip().lower()

    meta = graph.get("meta", {})
    fps = float(meta.get("fps", 1) or 1)
    normalized = []

    if "relations_seconds" in graph:
        merged_by_edge = {}
        for relation in graph.get("relations_seconds", []):
            if not isinstance(relation, (list, tuple)) or len(relation) != 4:
                continue
            subject_id, object_id, predicate, spans = relation
            key = (subject_id, object_id, _norm_predicate(predicate))
            if key not in merged_by_edge:
                merged_by_edge[key] = []
            for span in (spans or []):
                if isinstance(span, (list, tuple)) and len(span) == 2:
                    merged_by_edge[key].append((float(span[0]), float(span[1])))
        for (subject_id, object_id, predicate), intervals in merged_by_edge.items():
            subject = object_labels.get(subject_id, "")
            obj = object_labels.get(object_id, "")
            normalized.append((subject, predicate, obj, intervals))
        return normalized

    if "relations" in graph:
        merged_by_edge = {}
        for relation in graph.get("relations", []):
            if not isinstance(relation, (list, tuple)) or len(relation) < 4:
                continue
            subject_id, object_id, predicate, spans = relation[:4]
            key = (subject_id, object_id, _norm_predicate(predicate))
            if key not in merged_by_edge:
                merged_by_edge[key] = []
            for span in (spans or []):
                if isinstance(span, (list, tuple)) and len(span) == 2:
                    merged_by_edge[key].append(
                        (float(span[0]) / fps, (float(span[1]) + 1) / fps)
                    )
        for (subject_id, object_id, predicate), intervals in merged_by_edge.items():
            subject = object_labels.get(subject_id, "")
            obj = object_labels.get(object_id, "")
            normalized.append((subject, predicate, obj, intervals))
        return normalized

    for relation in graph.get("relationships", []):
        if not isinstance(relation, dict):
            continue
        subject = relation.get("subject") or object_labels.get(relation.get("subject_id"), "")
        obj = relation.get("object") or object_labels.get(relation.get("object_id"), "")
        duration = relation.get("duration")
        intervals = []
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            intervals = [(float(duration[0]), float(duration[1]))]
        else:
            intervals = [
                (float(span[0]), float(span[1]))
                for span in relation.get("segments", [])
                if isinstance(span, (list, tuple)) and len(span) == 2
            ]
        normalized.append((
            str(subject).strip().lower(),
            _norm_predicate(relation.get("predicate", "")),
            str(obj).strip().lower(),
            intervals,
        ))
    return normalized


def compute_tiou(spans1, spans2):
    """Compute temporal IoU for possibly discontinuous half-open intervals."""
    def merge(spans):
        merged = []
        for start, end in sorted((float(a), float(b)) for a, b in spans if b > a):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    first = merge(spans1)
    second = merge(spans2)
    intersection = sum(
        max(0.0, min(end_a, end_b) - max(start_a, start_b))
        for start_a, end_a in first
        for start_b, end_b in second
    )
    total_a = sum(end - start for start, end in first)
    total_b = sum(end - start for start, end in second)
    union = total_a + total_b - intersection
    return intersection / union if union > 0 else 0.0


def select_ground_truth(gt_data: dict, video_id: str):
    if isinstance(gt_data, dict) and isinstance(gt_data.get("data"), list):
        for record in gt_data["data"]:
            if record.get("video_id") == video_id:
                return record
        raise ValueError(f"Video {video_id!r} is not present in the ground-truth dataset")
    if isinstance(gt_data, dict) and gt_data.get("video_id") not in (None, video_id):
        raise ValueError(
            f"Ground-truth video_id {gt_data['video_id']!r} does not match {video_id!r}"
        )
    return gt_data


_ALIGNMENT_LABELS = {
    "identical",
    "synonym",
    "hypernym/hyponym",
    "semantic overlap",
    "mismatch",
}


def _canonical_alignment_class(text: str) -> str:
    norm = str(text).strip().lower()
    if norm in _ALIGNMENT_LABELS:
        return norm
    clean = re.sub(r"\s*/\s*", "/", norm)
    clean = re.sub(r"\s+", " ", clean)
    if clean in _ALIGNMENT_LABELS:
        return clean
    if "identic" in clean:
        return "identical"
    if "synonym" in clean:
        return "synonym"
    if "hypernym" in clean or "hyponym" in clean:
        return "hypernym/hyponym"
    if "overlap" in clean:
        return "semantic overlap"
    return "mismatch"


def build_semantic_alignment(pred_graph, gt_graph, config, batch_size=40, cache_path=None):
    pred_labels = _normalize_graph(pred_graph)
    gt_labels = _normalize_graph(gt_graph)
    pairs = set()
    pred_objects = _normalize_object_labels(pred_graph)
    gt_objects = _normalize_object_labels(gt_graph)
    pairs.update(
        ("object", gt_label, pred_label)
        for gt_label in gt_objects
        for pred_label in pred_objects
    )
    for gt in gt_labels:
        for pred in pred_labels:
            pairs.update((
                ("object", gt[0], pred[0]),
                ("predicate", gt[1], pred[1]),
                ("object", gt[2], pred[2]),
            ))

    alignments = {}
    pending = []
    for pair in sorted(pairs):
        if pair[1] == pair[2]:
            alignments[pair] = "identical"
        else:
            pending.append(pair)

    # Persistent cache so repeated (element_type, gt, pred) pairs across videos
    # do not trigger duplicate LLM calls.
    cache = {}
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            try:
                for row in json.loads(cache_path.read_text(encoding="utf-8")):
                    cache[(row["element_type"], row["ground_truth"], row["prediction"])] = row["class"]
            except Exception:
                cache = {}
    still_pending = []
    for pair in pending:
        if pair in cache and cache[pair] in _ALIGNMENT_LABELS:
            alignments[pair] = cache[pair]
        else:
            still_pending.append(pair)
    pending = still_pending

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        pair_rows = [
            {
                "id": index,
                "element_type": kind,
                "ground_truth": ground_truth_label,
                "prediction": predicted_label,
            }
            for index, (kind, ground_truth_label, predicted_label) in enumerate(batch)
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a pairwise lexical semantic matcher. For each row you get two short labels "
                    "and the category they belong to (object or predicate). Judge ONLY what the two labels "
                    "mean as words or phrases. Ignore any video, scene, or task context. Do not judge which "
                    "label is more accurate; judge whether the two labels can refer to the same thing.\n\n"
                    "Decide the class by applying these tests IN ORDER, stopping at the first test that fires and be lenient:\n"
                    "1. IDENTICAL — the two labels have the same meaning. Surface form, singular/plural, "
                    "spelling, or word order may differ (e.g. 'sofa' vs 'couch', 'people' vs 'person').\n"
                    "2. SYNONYM — the labels are interchangeable in ordinary usage: substituting one for the "
                    "other would not change what is being described. Multi-word phrases that differ only by a "
                    "posture or motion modifier before the same core "
                    "relation describe the same relation, so they are SYNONYM.\n"
                    "3. HYPERNYM/HYPONYM — one label is a subtype, role, or generalization of the other: "
                    "an instance of one is necessarily an instance of the other. This covers role words for "
                    "people (a 'football player' IS an adult person; a 'woman' IS an adult), species, vehicle "
                    "types, and container/furniture types.\n"
                    "4. SEMANTIC OVERLAP — the meanings share substantial content but neither test above "
                    "fires: same domain, same broad category, or same action family, yet not interchangeable "
                    "and not subtype-related.\n"
                    "5. MISMATCH — none of the above: different meanings, unrelated domains, or opposite "
                    "relations.\n\n"
                    "Guardrails:\n"
                    "- A person described by their activity or role (player, coach, rider, driver, worker) is "
                    "HYPERNYM/HYPONYM with the general person label, never MISMATCH.\n"
                    "- Two labels for the same physical setting (a floor covering, a seat furniture, a "
                    "cooking surface) are at least SEMANTIC OVERLAP, usually SYNONYM.\n"
                    "- A general spatial relation and its specific posture variant (being on a surface vs "
                    "standing/sitting on it) are SYNONYM, not MISMATCH.\n"
                    "- Opposites ('on' vs 'under', 'in front of' vs 'behind') are MISMATCH.\n"
                    "- Do not invent relationships between words that merely co-occur in scenes.\n\n"
                    "Choose exactly one class per row: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, "
                    "or Mismatch. Return only JSON with shape {\"matches\":[{\"id\":integer,\"class\":string}]} "
                    "and include every input id exactly once. Output nothing except the JSON."
                ),
            },
            {"role": "user", "content": json.dumps({"pairs": pair_rows}, ensure_ascii=False)},
        ]
        response = call_llm_api(
            messages=messages,
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_key=config.get("api_key"),
            api_version=config.get("api_version"),
            max_tokens=max(2048, len(batch) * 64),
            seed=config.get("seed", 42),
            temperature=0,
            return_json=True,
        )
        raw = get_message_content(response)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except (TypeError, json.JSONDecodeError):
            match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except Exception as exc:
                    raise RuntimeError(f"Semantic aligner returned invalid JSON: {raw[:500]!r}") from exc
            else:
                raise RuntimeError(f"Semantic aligner returned invalid JSON: {raw[:500]!r}")

        returned = None
        if isinstance(parsed, list):
            returned = parsed
        elif isinstance(parsed, dict):
            # Check for single item match: {'id': 0, 'class': 'Mismatch'}
            if "id" in parsed or "class" in parsed or "class_name" in parsed:
                returned = [parsed]
            else:
                for k in ("matches", "pairs", "results", "alignments", "classifications", "data"):
                    if isinstance(parsed.get(k), list):
                        returned = parsed[k]
                        break
                if returned is None:
                    # Check if it's a dict mapping id string to class: {"0": "Mismatch", "1": "Synonym"}
                    if parsed and all(str(k).isdigit() for k in parsed.keys()):
                        returned = [{"id": int(k), "class": str(v)} for k, v in parsed.items()]
                    else:
                        for v in parsed.values():
                            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                                returned = v
                                break

        if not isinstance(returned, list):
            raise RuntimeError(f"Semantic aligner response must contain a 'matches' array or list, got: {str(parsed)[:300]}")

        for item in returned:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id", item.get("pair_id", item.get("row_id", 0)))
            try:
                index = int(raw_id)
            except Exception:
                continue
            if 0 <= index < len(batch):
                raw_class = item.get("class", item.get("class_name", item.get("label", item.get("alignment", "mismatch"))))
                label = _canonical_alignment_class(raw_class)
                alignments[batch[index]] = label

        for pair in batch:
            alignments.setdefault(pair, "mismatch")

    if cache_path is not None:
        if cache_path.is_file():
            try:
                for row in json.loads(cache_path.read_text(encoding="utf-8")):
                    cache[(row["element_type"], row["ground_truth"], row["prediction"])] = row["class"]
            except Exception:
                pass
        cache.update({pair: label for pair, label in alignments.items()})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_cache = cache_path.with_suffix(f".tmp_{os.getpid()}")
        try:
            temp_cache.write_text(
                json.dumps(
                    [
                        {"element_type": k[0], "ground_truth": k[1], "prediction": k[2], "class": v}
                        for k, v in sorted(cache.items())
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            temp_cache.replace(cache_path)
        except Exception:
            pass

    return alignments


def _score_relation_matches(pred_rels, gt_rels, threshold, semantic_alignment=None):
    def labels_match(gt, pred):
        if semantic_alignment is None:
            return gt[:3] == pred[:3]
        component_kinds = ("object", "predicate", "object")
        return all(
            semantic_alignment.get((kind, gt_label, pred_label), "mismatch") != "mismatch"
            for kind, gt_label, pred_label in zip(component_kinds, gt[:3], pred[:3])
        )

    adjacency = {}
    for gt_index, gt in enumerate(gt_rels):
        overlaps = []
        for pred_index, pred in enumerate(pred_rels):
            if labels_match(gt, pred):
                tiou = compute_tiou(gt[3], pred[3])
                if tiou >= threshold:
                    overlaps.append((tiou, pred_index))
        adjacency[gt_index] = [pred_index for _, pred_index in sorted(overlaps, reverse=True)]

    pred_to_gt = {}

    def assign(gt_index, visited_predictions):
        for pred_index in adjacency[gt_index]:
            if pred_index in visited_predictions:
                continue
            visited_predictions.add(pred_index)
            if pred_index not in pred_to_gt or assign(pred_to_gt[pred_index], visited_predictions):
                pred_to_gt[pred_index] = gt_index
                return True
        return False

    return sum(assign(gt_index, set()) for gt_index in range(len(gt_rels)))


def _normalize_object_labels(graph):
    objects = graph.get("objects", graph.get("entities", []))
    labels = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        label = item.get("category", item.get("label"))
        if label is not None:
            labels.append(str(label).strip().lower())
    return labels


def _score_object_labels(pred_objects, gt_objects, semantic_alignment=None):
    adjacency = {}
    for gt_index, gt_label in enumerate(gt_objects):
        adjacency[gt_index] = [
            pred_index
            for pred_index, pred_label in enumerate(pred_objects)
            if (
                gt_label == pred_label
                if semantic_alignment is None
                else semantic_alignment.get(("object", gt_label, pred_label), "mismatch")
                != "mismatch"
            )
        ]

    pred_to_gt = {}

    def assign(gt_index, visited_predictions):
        for pred_index in adjacency[gt_index]:
            if pred_index in visited_predictions:
                continue
            visited_predictions.add(pred_index)
            if pred_index not in pred_to_gt or assign(
                pred_to_gt[pred_index], visited_predictions
            ):
                pred_to_gt[pred_index] = gt_index
                return True
        return False

    matched = sum(assign(gt_index, set()) for gt_index in range(len(gt_objects)))
    return {
        "matched": matched,
        "precision": matched / len(pred_objects) if pred_objects else 0.0,
        "recall": matched / len(gt_objects) if gt_objects else 0.0,
    }


def evaluate_scene_graph(
    pred_graph: dict,
    gt_graph: dict,
    tiou_thresholds=(0.3, 0.5),
    semantic_alignment=None,
):
    pred_rels = _normalize_graph(pred_graph)
    gt_rels = _normalize_graph(gt_graph)

    print("\n========================================")
    print("EVALUATION RESULTS vs GROUND TRUTH")
    print(f"Predicted Triplets: {len(pred_rels)} | GT Triplets: {len(gt_rels)}")
    print("========================================")

    pred_objects = _normalize_object_labels(pred_graph)
    gt_objects = _normalize_object_labels(gt_graph)
    strict_objects = _score_object_labels(pred_objects, gt_objects)
    results = {"object_strict": strict_objects}
    print(
        "Strict object labels: "
        f"{strict_objects['matched']} matches; "
        f"precision={strict_objects['precision'] * 100:.2f}%, "
        f"recall={strict_objects['recall'] * 100:.2f}%"
    )

    scoring_modes = [("strict", None)]
    if semantic_alignment is not None:
        scoring_modes.append(("lenient", semantic_alignment))

    for mode, alignment in scoring_modes:
        print(f"\n{mode.capitalize()} label matching")
        if mode == "lenient":
            lenient_objects = _score_object_labels(
                pred_objects, gt_objects, semantic_alignment
            )
            results["object_lenient"] = lenient_objects
            print(
                f"Object labels: {lenient_objects['matched']} matches; "
                f"precision={lenient_objects['precision'] * 100:.2f}%, "
                f"recall={lenient_objects['recall'] * 100:.2f}%"
            )
        for threshold in tiou_thresholds:
            matched = _score_relation_matches(pred_rels, gt_rels, threshold, alignment)
            recall = matched / len(gt_rels) if gt_rels else 0.0
            precision = matched / len(pred_rels) if pred_rels else 0.0
            metrics = {
                "matched": matched,
                "precision": precision,
                "recall": recall,
            }
            if mode == "strict":
                results[threshold] = metrics
            else:
                results.setdefault("lenient", {})[threshold] = metrics
            print(
                f"tIoU>={threshold}: {matched} matches; "
                f"precision={precision * 100:.2f}%, recall={recall * 100:.2f}%"
            )

    return results


def build_relation_diagnostics(pred_graph, gt_graph, tiou_thresholds):
    """For each GT relation, find the best same-label predicted triplet and its
    tIoU, separating lexical failures from temporal failures."""
    pred_rels = _normalize_graph(pred_graph)
    gt_rels = _normalize_graph(gt_graph)
    rows = []
    for gt in gt_rels:
        best_tiou = 0.0
        best_pred = None
        for pred in pred_rels:
            if gt[:3] == pred[:3]:
                tiou = compute_tiou(gt[3], pred[3])
                if tiou > best_tiou:
                    best_tiou = tiou
                    best_pred = [pred[0], pred[1], pred[2], [[a, b] for a, b in pred[3]]]
        rows.append({
            "ground_truth": {"subject": gt[0], "predicate": gt[1], "object": gt[2], "spans": [[a, b] for a, b in gt[3]]},
            "best_exact_label_prediction": best_pred,
            "best_tiou": best_tiou,
            "passed": {str(t): best_tiou >= t for t in tiou_thresholds},
            "failure": (
                "none" if best_pred and best_tiou >= max(tiou_thresholds)
                else "temporal" if best_pred
                else "lexical_or_missing"
            ),
        })
    return rows


def evaluate_single_video(
    video_path: Path,
    args,
    config_base: dict,
    gt_data: dict,
    object_categories: dict,
    predicates: list,
    output_dir: Path,
    judge_config: dict = None,
):
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    gt_record = None
    if gt_data:
        try:
            gt_record = select_ground_truth(gt_data, video_path.stem)
        except ValueError as e:
            print(f"Warning: {e}")
            gt_record = None

    config = dict(config_base)
    config["scenegraph_fps"] = (
        gt_record.get("meta", {}).get("fps", args.annotation_fps)
        if gt_record
        else args.annotation_fps
    )

    print(f"\n========================================")
    print(f"Running SceneGraphAgent on: {video_path.name}")
    print(f"========================================")

    agent = SceneGraphAgent(
        config=config,
        video_path=str(video_path),
        subtitle_path=args.subtitle_path,
        output_dir=str(output_dir),
        verbose=args.verbose,
    )

    t0 = time.time()
    traj = agent.run("Generate a complete spatio-temporal scene graph for this video.")
    elapsed = time.time() - t0

    traj_dict = traj.to_dict()
    raw_answer = traj_dict.get("final_answer", "")

    video_id = video_path.stem
    run_dir = output_dir / video_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectory.json").write_text(
        json.dumps(traj_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "answer_raw.txt").write_text(raw_answer, encoding="utf-8")

    pred_graph = None
    try:
        pred_graph = json.loads(raw_answer)
    except Exception:
        try:
            pred_graph = repair_and_parse_scenegraph_json(raw_answer)
        except Exception:
            pred_graph = None

    if (
        not isinstance(pred_graph, dict)
        or not isinstance(pred_graph.get("objects"), list)
        or not isinstance(pred_graph.get("relations_seconds"), list)
    ):
        excerpt = raw_answer[:1200].replace("\n", " ")
        raise ValueError(
            "Scene-graph answer is not valid open-vocabulary JSON with 'objects' and 'relations_seconds' fields. "
            f"Raw answer saved to {run_dir / 'answer_raw.txt'}; excerpt: {excerpt!r}"
        )

    pred_graph["video_id"] = video_path.stem
    pred_graph["meta"] = agent.video_meta
    (run_dir / "scenegraph_raw.json").write_text(
        json.dumps(pred_graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if object_categories and predicates:
        pvsg_graph = to_pvsg_graph(pred_graph, object_categories, predicates)
        (run_dir / "scenegraph.json").write_text(
            json.dumps(pvsg_graph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"PVSG export saved to: {run_dir / 'scenegraph.json'}")

    print(f"Scene graph generation finished in {elapsed:.1f}s (saved to {run_dir})")

    evaluation = None
    if gt_record:
        semantic_alignment = None
        if args.lenient_semantic:
            try:
                active_judge_config = judge_config or config
                semantic_alignment = build_semantic_alignment(
                    pred_graph,
                    gt_record,
                    active_judge_config,
                    cache_path=output_dir / "semantic_alignment_cache.json",
                )
                alignment_counts = Counter(semantic_alignment.values())
                print(f"Pairwise lexical judgments: {len(semantic_alignment)}")
                print(f"Alignment classes: {dict(alignment_counts)}")
                (run_dir / "semantic_alignment.json").write_text(
                    json.dumps(
                        [
                            {
                                "element_type": element_type,
                                "ground_truth": ground_truth_label,
                                "prediction": predicted_label,
                                "class": alignment,
                            }
                            for (element_type, ground_truth_label, predicted_label), alignment
                            in sorted(semantic_alignment.items())
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                # Lenient scoring must not prevent strict evaluation.
                print(f"Warning: lenient semantic alignment failed ({e}); scoring strict only.")
                semantic_alignment = None
        tiou_thresholds = tuple(
            float(x.strip()) for x in args.tiou_thresholds.split(",") if x.strip()
        )
        evaluation = evaluate_scene_graph(
            pred_graph,
            gt_record,
            tiou_thresholds=tiou_thresholds,
            semantic_alignment=semantic_alignment,
        )
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "relation_diagnostics.json").write_text(
            json.dumps(
                build_relation_diagnostics(pred_graph, gt_record, tiou_thresholds),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return {"video_id": video_id, "elapsed": elapsed, "evaluation": evaluation, "run_dir": str(run_dir)}


def evaluate_saved_predictions(
    pred_dir: Path,
    args,
    gt_data: dict,
    object_categories: dict,
    predicates: list,
    output_dir: Path,
    judge_config: dict = None,
):
    """Score saved scenegraph_raw.json files without re-running the agent."""
    if judge_config is None:
        judge_config = dict(general_config)
        judge_config.update({
            "model_name": args.judge_model_name or args.model_name,
            "api_base": args.judge_api_base or args.api_base,
            "api_key": args.judge_api_key or args.api_key or os.environ.get("NVIDIA_NIM_API") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "api_version": None,
            "seed": 42,
            "temperature": 0,
            "scenegraph_object_categories": object_categories,
            "scenegraph_predicates": predicates,
        })

    tiou_thresholds = tuple(
        float(x.strip()) for x in args.tiou_thresholds.split(",") if x.strip()
    )
    results = []
    run_dirs = sorted(p for p in pred_dir.iterdir() if p.is_dir())
    if args.split and args.split != "all" and gt_data and isinstance(gt_data.get("split"), dict):
        target_ds = args.dataset_name or "vidor"
        ds_splits = gt_data["split"].get(target_ds, gt_data["split"])
        split_video_ids = set(ds_splits.get(args.split, []))
        if split_video_ids:
            run_dirs = [p for p in run_dirs if p.name in split_video_ids]
    for run_dir in run_dirs:
        raw_path = run_dir / "scenegraph_raw.json"
        if not raw_path.is_file():
            continue
        video_id = run_dir.name
        try:
            gt_record = select_ground_truth(gt_data, video_id)
        except ValueError as e:
            print(f"Skipping {video_id}: {e}")
            continue
        pred_graph = json.loads(raw_path.read_text(encoding="utf-8"))

        semantic_alignment = None
        if args.lenient_semantic:
            try:
                semantic_alignment = build_semantic_alignment(
                    pred_graph,
                    gt_record,
                    judge_config,
                    cache_path=output_dir / "semantic_alignment_cache.json",
                )
                print(f"{video_id} alignment classes: {dict(Counter(semantic_alignment.values()))}")
                (run_dir / "semantic_alignment.json").write_text(
                    json.dumps(
                        [
                            {
                                "element_type": element_type,
                                "ground_truth": ground_truth_label,
                                "prediction": predicted_label,
                                "class": alignment,
                            }
                            for (element_type, ground_truth_label, predicted_label), alignment
                            in sorted(semantic_alignment.items())
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"Warning: lenient alignment failed for {video_id} ({e}); strict only.")

        evaluation = evaluate_scene_graph(
            pred_graph,
            gt_record,
            tiou_thresholds=tiou_thresholds,
            semantic_alignment=semantic_alignment,
        )
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "relation_diagnostics.json").write_text(
            json.dumps(
                build_relation_diagnostics(pred_graph, gt_record, tiou_thresholds),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        results.append({"video_id": video_id, "evaluation": evaluation, "run_dir": str(run_dir)})

    return results, tiou_thresholds


def main():
    args = parse_args()
    if not args.video_path and not args.video_dir and not args.pred_dir:
        raise ValueError("Please provide --video_path, --video_dir, or --pred_dir (eval-only mode).")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("NVIDIA_NIM_API") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    gt_data = None
    if args.gt_file:
        gt_path = Path(args.gt_file).expanduser().resolve()
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
        with gt_path.open("r", encoding="utf-8") as f:
            gt_data = json.load(f)

    taxonomy_path = (
        Path(args.taxonomy_file).expanduser().resolve()
        if args.taxonomy_file
        else Path(__file__).resolve().parent / "svg2_dataset" / "PVSG dataset.json"
    )
    if args.taxonomy_file and not taxonomy_path.is_file():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
    taxonomy_data = None
    if args.taxonomy_file and taxonomy_path.is_file():
        with taxonomy_path.open("r", encoding="utf-8") as f:
            taxonomy_data = json.load(f)
    elif isinstance(gt_data, dict) and isinstance(gt_data.get("data"), list):
        taxonomy_data = gt_data
    elif taxonomy_path.is_file():
        with taxonomy_path.open("r", encoding="utf-8") as f:
            taxonomy_data = json.load(f)

    object_categories = taxonomy_data.get("objects", {}) if taxonomy_data else {}
    predicates = taxonomy_data.get("relations", []) if taxonomy_data else []

    judge_model_name = args.judge_model_name or args.model_name
    judge_api_base = args.judge_api_base or args.api_base
    judge_api_key = (
        args.judge_api_key
        or api_key
        or os.environ.get("NVIDIA_NIM_API")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    judge_config = dict(general_config)
    judge_config.update({
        "model_name": judge_model_name,
        "api_base": judge_api_base,
        "api_key": judge_api_key,
        "api_version": None,
        "seed": 42,
        "temperature": 0,
        "scenegraph_object_categories": object_categories,
        "scenegraph_predicates": predicates,
    })

    # ---- Eval-only mode: score saved predictions, no agent run ----
    if args.pred_dir:
        if gt_data is None:
            raise ValueError("--pred_dir requires --gt_file for metric calculation.")
        pred_dir = Path(args.pred_dir).expanduser().resolve()
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
        print(f"\n========================================")
        print(f"SCENE GRAPH EVALUATION (EVAL-ONLY): {pred_dir}")
        print(f"Agent model: {args.model_name}")
        if args.lenient_semantic:
            print(f"Judge model: {judge_config['model_name']}")
        print(f"Output directory: {output_dir}")
        print(f"========================================")
        results, tiou_thresholds = evaluate_saved_predictions(
            pred_dir, args, gt_data, object_categories, predicates, output_dir, judge_config=judge_config
        )
        successful = [r for r in results if r.get("evaluation") is not None]
        if len(successful) > 1:
            n = len(successful)
            summary = {
                "total_videos": len(results),
                "successful_evaluations": n,
                "agent_model": args.model_name,
                "judge_model": judge_config["model_name"] if args.lenient_semantic else None,
                "object_strict": {
                    "precision": sum(r["evaluation"]["object_strict"]["precision"] for r in successful) / n,
                    "recall": sum(r["evaluation"]["object_strict"]["recall"] for r in successful) / n,
                },
                "relations_strict": {},
            }
            for thresh in tiou_thresholds:
                summary["relations_strict"][str(thresh)] = {
                    "precision": sum(r["evaluation"].get(thresh, {}).get("precision", 0.0) for r in successful) / n,
                    "recall": sum(r["evaluation"].get(thresh, {}).get("recall", 0.0) for r in successful) / n,
                }
            if args.lenient_semantic:
                summary["object_lenient"] = {
                    "precision": sum(r["evaluation"].get("object_lenient", {}).get("precision", 0.0) for r in successful) / n,
                    "recall": sum(r["evaluation"].get("object_lenient", {}).get("recall", 0.0) for r in successful) / n,
                }
                summary["relations_lenient"] = {}
                for thresh in tiou_thresholds:
                    summary["relations_lenient"][str(thresh)] = {
                        "precision": sum(r["evaluation"].get("lenient", {}).get(thresh, {}).get("precision", 0.0) for r in successful) / n,
                        "recall": sum(r["evaluation"].get("lenient", {}).get(thresh, {}).get("recall", 0.0) for r in successful) / n,
                    }
            (output_dir / "batch_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nSummary saved to: {output_dir / 'batch_summary.json'}")
        return

    config_base = dict(general_config)
    config_base.update({
        "model_name": args.model_name,
        "api_base": args.api_base,
        "api_key": api_key,
        "api_version": None,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "frame_sampling_factor": args.frame_sampling_factor,
        "reasoning_effort": None,
        "seed": 42,
        "temperature": 0.2,
        "scenegraph_object_categories": object_categories,
        "scenegraph_predicates": predicates,
    })

    # Collect videos
    if args.video_path:
        videos = [Path(args.video_path).expanduser().resolve()]
    else:
        video_dir = Path(args.video_dir).expanduser().resolve()
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        all_videos = sorted(
            [p for p in video_dir.iterdir() if p.suffix.lower() == ".mp4"],
            key=lambda p: p.name,
        )
        if args.split and args.split != "all" and gt_data and isinstance(gt_data.get("split"), dict):
            split_dict = gt_data["split"]
            target_ds = args.dataset_name or "vidor"
            ds_splits = split_dict.get(target_ds, split_dict)
            split_video_ids = set(ds_splits.get(args.split, []))
            if split_video_ids:
                videos = [v for v in all_videos if v.stem in split_video_ids]
                print(f"Filtered {len(videos)} video(s) for split={args.split!r} (dataset={target_ds!r})")
            else:
                print(f"Warning: No video IDs found for split={args.split!r} in {target_ds!r}")
                videos = all_videos
        elif gt_data and isinstance(gt_data.get("data"), list):
            gt_video_ids = {r["video_id"] for r in gt_data["data"]}
            videos = [v for v in all_videos if v.stem in gt_video_ids]
        else:
            videos = all_videos
        if args.num_samples:
            videos = videos[: args.num_samples]

    print(f"\n========================================")
    print(f"SCENE GRAPH EVALUATION: {len(videos)} video(s)")
    print(f"Agent model: {config_base['model_name']}")
    if args.lenient_semantic:
        print(f"Judge model: {judge_config['model_name']}")
    print(f"Output directory: {output_dir}")
    print(f"========================================")

    results = []
    tiou_thresholds = tuple(
        float(x.strip()) for x in args.tiou_thresholds.split(",") if x.strip()
    )

    for idx, v_path in enumerate(videos, start=1):
        print(f"\n>>> [{idx}/{len(videos)}] Processing: {v_path.name}")
        try:
            res = evaluate_single_video(
                v_path,
                args,
                config_base,
                gt_data,
                object_categories,
                predicates,
                output_dir,
                judge_config=judge_config,
            )
            results.append(res)
        except Exception as e:
            print(f"Error evaluating {v_path.name}: {e}")
            error_dir = output_dir / v_path.stem
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.txt").write_text(
                f"{type(e).__name__}: {e}\n", encoding="utf-8"
            )
            results.append({"video_id": v_path.stem, "error": str(e)})

    # Summary
    successful = [r for r in results if r.get("evaluation") is not None]
    if len(videos) > 1 and successful:
        n = len(successful)
        summary = {
            "total_videos": len(videos),
            "successful_evaluations": n,
            "agent_model": config_base["model_name"],
            "judge_model": judge_config["model_name"] if args.lenient_semantic else None,
            "object_strict": {
                "precision": sum(r["evaluation"]["object_strict"]["precision"] for r in successful) / n,
                "recall": sum(r["evaluation"]["object_strict"]["recall"] for r in successful) / n,
            },
            "relations_strict": {},
        }
        for thresh in tiou_thresholds:
            summary["relations_strict"][str(thresh)] = {
                "precision": sum(r["evaluation"].get(thresh, {}).get("precision", 0.0) for r in successful) / n,
                "recall": sum(r["evaluation"].get(thresh, {}).get("recall", 0.0) for r in successful) / n,
            }

        if args.lenient_semantic:
            summary["object_lenient"] = {
                "precision": sum(r["evaluation"].get("object_lenient", {}).get("precision", 0.0) for r in successful) / n,
                "recall": sum(r["evaluation"].get("object_lenient", {}).get("recall", 0.0) for r in successful) / n,
            }
            summary["relations_lenient"] = {}
            for thresh in tiou_thresholds:
                summary["relations_lenient"][str(thresh)] = {
                    "precision": sum(r["evaluation"].get("lenient", {}).get(thresh, {}).get("precision", 0.0) for r in successful) / n,
                    "recall": sum(r["evaluation"].get("lenient", {}).get(thresh, {}).get("recall", 0.0) for r in successful) / n,
                }

        (output_dir / "batch_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\n" + "=" * 55)
        print(f"BATCH EVALUATION SUMMARY ({n}/{len(videos)} videos)")
        print("=" * 55)
        print(f"Strict Object Match: Precision={summary['object_strict']['precision']*100:.2f}%, Recall={summary['object_strict']['recall']*100:.2f}%")
        if args.lenient_semantic:
            print(f"Lenient Object Match: Precision={summary['object_lenient']['precision']*100:.2f}%, Recall={summary['object_lenient']['recall']*100:.2f}%")
        print("-" * 55)
        for thresh in tiou_thresholds:
            s_p = summary["relations_strict"][str(thresh)]["precision"] * 100
            s_r = summary["relations_strict"][str(thresh)]["recall"] * 100
            line = f"Strict Rel tIoU>={thresh}: Precision={s_p:.2f}%, Recall={s_r:.2f}%"
            if args.lenient_semantic:
                l_p = summary["relations_lenient"][str(thresh)]["precision"] * 100
                l_r = summary["relations_lenient"][str(thresh)]["recall"] * 100
                line += f" | Lenient: Precision={l_p:.2f}%, Recall={l_r:.2f}%"
            print(line)
        print("=" * 55)
        print(f"Summary saved to: {output_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
