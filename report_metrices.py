#!/usr/bin/env python3
"""Report scene-graph metrics on saved runs. No model calls are made.

Metrics (per video and aggregated)
----------------------------------
- triplet_recall  : category-level. GT relations are deduplicated by
  (subject_category, predicate, object_category). A triplet is recalled at
  threshold t if some predicted relation with matching labels has tIoU >= t
  against any GT entry of that triplet.
- relation_recall : instance-level. Each GT relation entry (one annotated
  instance with its spans) must be matched one-to-one by a predicted relation
  with matching labels and tIoU >= t (maximum bipartite matching).
- object_accuracy : one-to-one matched predicted objects / total predicted
  objects (classification accuracy of the predictions).
- object_recall   : one-to-one matched predicted objects / total GT objects.
- attribute_recall: label-level recall of GT object attributes/status.
  Reported as n/a when the GT subset has no attribute annotations (the VidOR
  subset of PVSG has none).

Label matching criteria
-----------------------
- strict : exact lowercase equality on (subject, predicate, object).
- lenient: reuses the saved semantic_alignment.json in each run directory;
  any class other than "mismatch" counts as a match. Pairs absent from the
  saved file are treated as mismatch (a warning count is printed).

Temporal matching: tIoU over merged half-open second spans, thresholds 0.1/0.3/0.5.

Usage:
    python report_metrics.py --pred_dir output/scenegraph_test_kimi \
        --gt_file "svg2_dataset/PVSG dataset.json"
"""

import argparse
import json
import re
from pathlib import Path

DEFAULT_TIOU = "0.1,0.3,0.5"


def _norm_predicate(predicate) -> str:
    """Canonicalize predicate text: underscores/hyphens become spaces so that
    'in_front_of' matches the PVSG predicate 'in front of'."""
    return re.sub(r"[_\-\s]+", " ", str(predicate)).strip().lower()


# ---------------------------------------------------------------- graph normalization

def _normalize_graph(graph: dict):
    """Return list of (subject_cat, predicate, object_cat, [(s, e), ...]) with
    lowercase labels and half-open second intervals."""
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
            merged_by_edge.setdefault(key, [])
            for span in (spans or []):
                if isinstance(span, (list, tuple)) and len(span) == 2:
                    merged_by_edge[key].append((float(span[0]), float(span[1])))
        for (subject_id, object_id, predicate), intervals in merged_by_edge.items():
            normalized.append((
                object_labels.get(subject_id, ""),
                predicate,
                object_labels.get(object_id, ""),
                intervals,
            ))
        return normalized

    if "relations" in graph:
        merged_by_edge = {}
        for relation in graph.get("relations", []):
            if not isinstance(relation, (list, tuple)) or len(relation) < 4:
                continue
            subject_id, object_id, predicate, spans = relation[:4]
            key = (subject_id, object_id, _norm_predicate(predicate))
            merged_by_edge.setdefault(key, [])
            for span in (spans or []):
                if isinstance(span, (list, tuple)) and len(span) == 2:
                    merged_by_edge[key].append(
                        (float(span[0]) / fps, (float(span[1]) + 1) / fps)
                    )
        for (subject_id, object_id, predicate), intervals in merged_by_edge.items():
            normalized.append((
                object_labels.get(subject_id, ""),
                predicate,
                object_labels.get(object_id, ""),
                intervals,
            ))
        return normalized

    for relation in graph.get("relationships", []):
        if not isinstance(relation, dict):
            continue
        subject = relation.get("subject") or object_labels.get(relation.get("subject_id"), "")
        obj = relation.get("object") or object_labels.get(relation.get("object_id"), "")
        duration = relation.get("duration")
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


def _normalize_object_labels(graph: dict):
    labels = []
    for item in graph.get("objects", graph.get("entities", [])):
        if not isinstance(item, dict):
            continue
        label = item.get("category", item.get("label"))
        if label is not None:
            labels.append(str(label).strip().lower())
    return labels


def compute_tiou(spans1, spans2):
    """Temporal IoU for possibly discontinuous half-open intervals."""
    def merge(spans):
        merged = []
        for start, end in sorted((float(a), float(b)) for a, b in spans if b > a):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    first, second = merge(spans1), merge(spans2)
    intersection = sum(
        max(0.0, min(ea, eb) - max(sa, sb))
        for sa, ea in first
        for sb, eb in second
    )
    total_a = sum(e - s for s, e in first)
    total_b = sum(e - s for s, e in second)
    union = total_a + total_b - intersection
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------- matching

def _max_match(adjacency, n_left):
    """Maximum bipartite matching (DFS augmenting paths)."""
    pred_to_gt = {}

    def assign(left_index, visited):
        for right_index in adjacency[left_index]:
            if right_index in visited:
                continue
            visited.add(right_index)
            if right_index not in pred_to_gt or assign(pred_to_gt[right_index], visited):
                pred_to_gt[right_index] = left_index
                return True
        return False

    return sum(assign(i, set()) for i in range(n_left))


def _make_label_matcher(alignment):
    if alignment is None:
        return lambda gt, pred: gt[:3] == pred[:3]
    kinds = ("object", "predicate", "object")

    def match(gt, pred):
        return all(
            alignment.get((kind, g, p), "mismatch") != "mismatch"
            for kind, g, p in zip(kinds, gt[:3], pred[:3])
        )

    return match


def relation_recall(pred_rels, gt_rels, threshold, label_match):
    """Instance-level: one-to-one matched GT relation entries / total GT entries."""
    adjacency = {}
    for gi, gt in enumerate(gt_rels):
        hits = []
        for pi, pred in enumerate(pred_rels):
            if label_match(gt, pred):
                tiou = compute_tiou(gt[3], pred[3])
                if tiou >= threshold:
                    hits.append((tiou, pi))
        adjacency[gi] = [pi for _, pi in sorted(hits, reverse=True)]
    return _max_match(adjacency, len(gt_rels))


def triplet_recall(pred_rels, gt_rels, threshold, label_match):
    """Category-level: recalled unique (s, p, o) triplets / total unique triplets."""
    by_triplet = {}
    for gt in gt_rels:
        by_triplet.setdefault(gt[:3], []).append(gt)
    recalled = 0
    for entries in by_triplet.values():
        if any(
            label_match(gt, pred) and compute_tiou(gt[3], pred[3]) >= threshold
            for gt in entries
            for pred in pred_rels
        ):
            recalled += 1
    return recalled, len(by_triplet)

def object_metrics(pred_objects, gt_objects, alignment):
    def ok(gt_label, pred_label):
        if alignment is None:
            return gt_label == pred_label
        return alignment.get(("object", gt_label, pred_label), "mismatch") != "mismatch"

    adjacency = {
        gi: [pi for pi, pl in enumerate(pred_objects) if ok(gl, pl)]
        for gi, gl in enumerate(gt_objects)
    }
    matched = _max_match(adjacency, len(gt_objects))
    return {
        "matched": matched,
        "accuracy": matched / len(pred_objects) if pred_objects else 0.0,
        "recall": matched / len(gt_objects) if gt_objects else 0.0,
    }


def attribute_recall(pred_graph, gt_record, alignment):
    """Label-level recall of GT object attributes/status. None => n/a."""

    def attrs(obj):
        val = obj.get("status") or obj.get("attributes") or []
        if isinstance(val, str):
            val = [val]
        return [str(a).strip().lower() for a in val if str(a).strip()]

    gt_pairs = set()
    for obj in gt_record.get("objects", []):
        if isinstance(obj, dict):
            cat = str(obj.get("category", "")).strip().lower()
            for a in attrs(obj):
                gt_pairs.add((cat, a))
    if not gt_pairs:
        return None

    pred_pairs = set()
    for obj in pred_graph.get("objects", []):
        if isinstance(obj, dict):
            cat = str(obj.get("category", obj.get("label", ""))).strip().lower()
            for a in attrs(obj):
                pred_pairs.add((cat, a))

    def cat_ok(gt_cat, pred_cat):
        if gt_cat == pred_cat:
            return True
        return alignment is not None and (
            alignment.get(("object", gt_cat, pred_cat), "mismatch") != "mismatch"
        )

    matched = sum(
        1
        for gt_cat, gt_attr in gt_pairs
        if any(cat_ok(gt_cat, pred_cat) and pred_attr == gt_attr for pred_cat, pred_attr in pred_pairs)
    )
    return matched / len(gt_pairs)


# ---------------------------------------------------------------- alignment loading

def load_alignment(run_dir: Path):
    """Rebuild the alignment dict from a saved semantic_alignment.json (no LLM)."""
    path = run_dir / "semantic_alignment.json"
    if not path.is_file():
        return None
    alignment = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        alignment[(row["element_type"], row["ground_truth"], row["prediction"])] = row["class"]
    return alignment


# ---------------------------------------------------------------- reporting

def evaluate_run_dir(run_dir: Path, gt_record: dict, thresholds):
    pred_path = run_dir / "scenegraph_raw.json"
    pred_graph = json.loads(pred_path.read_text(encoding="utf-8"))
    pred_rels = _normalize_graph(pred_graph)
    gt_rels = _normalize_graph(gt_record)
    pred_objects = _normalize_object_labels(pred_graph)
    gt_objects = _normalize_object_labels(gt_record)

    alignment = load_alignment(run_dir)
    modes = [("strict", None)]
    if alignment is not None:
        modes.append(("lenient", alignment))

    rows = []
    for mode, align in modes:
        label_match = _make_label_matcher(align)
        obj = object_metrics(pred_objects, gt_objects, align)
        attr = attribute_recall(pred_graph, gt_record, align)
        for threshold in thresholds:
            trip_matched, trip_total = triplet_recall(pred_rels, gt_rels, threshold, label_match)
            rel_matched = relation_recall(pred_rels, gt_rels, threshold, label_match)
            rows.append({
                "criterion": mode,
                "tiou": threshold,
                "triplet_matched": trip_matched,
                "triplet_total": trip_total,
                "triplet_recall": trip_matched / trip_total if trip_total else 0.0,
                "relation_matched": rel_matched,
                "relation_total": len(gt_rels),
                "relation_recall": rel_matched / len(gt_rels) if gt_rels else 0.0,
                "object_accuracy": obj["accuracy"],
                "object_recall": obj["recall"],
                "attribute_recall": attr,
            })
    meta = {
        "gt_relations": len(gt_rels),
        "pred_relations": len(pred_rels),
        "gt_objects": len(gt_objects),
        "pred_objects": len(pred_objects),
        "lenient_available": alignment is not None,
    }
    return rows, meta


def print_table(rows, title):
    print(f"\n{title}")
    header = f"{'criterion':<10} {'tIoU':>5} {'triplet_R':>10} {'relation_R':>11} {'obj_acc':>8} {'obj_rec':>8} {'attr_R':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        attr = "n/a" if row["attribute_recall"] is None else f"{row['attribute_recall'] * 100:6.1f}%"
        print(
            f"{row['criterion']:<10} {row['tiou']:>5.1f} "
            f"{row['triplet_matched']:>4}/{row['triplet_total']:<3}{row['triplet_recall'] * 100:5.1f}% "
            f"{row['relation_matched']:>4}/{row['relation_total']:<3}{row['relation_recall'] * 100:5.1f}% "
            f"{row['object_accuracy'] * 100:7.1f}% {row['object_recall'] * 100:7.1f}% {attr:>7}"
        )


def main():
    ap = argparse.ArgumentParser(description="Report saved scene-graph metrics (no model calls).")
    ap.add_argument("--pred_dir", required=True, help="Directory of run subdirs containing scenegraph_raw.json.")
    ap.add_argument("--gt_file", default="svg2_dataset/PVSG dataset.json", help="PVSG dataset JSON with ground truth.")
    ap.add_argument("--tiou_thresholds", default=DEFAULT_TIOU, help="Comma-separated tIoU thresholds.")
    ap.add_argument("--split", default=None, choices=["val", "train", "all"], help="Filter by split (e.g. 'val' or 'train').")
    ap.add_argument("--dataset_name", default="vidor", help="Dataset name inside ground-truth split dict (default: 'vidor').")
    ap.add_argument("--output_json", default=None, help="Optional path for the full JSON report.")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    thresholds = tuple(float(x.strip()) for x in args.tiou_thresholds.split(",") if x.strip())

    gt_path = Path(args.gt_file).expanduser().resolve()
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_index = {record["video_id"]: record for record in gt_data.get("data", [])}

    split_filter_ids = None
    if args.split and args.split != "all" and isinstance(gt_data.get("split"), dict):
        target_ds = args.dataset_name or "vidor"
        ds_splits = gt_data["split"].get(target_ds, gt_data["split"])
        split_filter_ids = set(ds_splits.get(args.split, []))
        if split_filter_ids:
            print(f"Filtering reporting to split={args.split!r} ({len(split_filter_ids)} target videos)")

    all_results = {}
    missing_lenient = []
    for run_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
        if not (run_dir / "scenegraph_raw.json").is_file():
            continue
        video_id = run_dir.name
        if split_filter_ids is not None and video_id not in split_filter_ids:
            continue
        gt_record = gt_index.get(video_id)
        if gt_record is None:
            print(f"Skipping {video_id}: no ground-truth record")
            continue
        rows, meta = evaluate_run_dir(run_dir, gt_record, thresholds)
        all_results[video_id] = {"meta": meta, "rows": rows}
        note = "" if meta["lenient_available"] else "  [lenient: no saved semantic_alignment.json]"
        print_table(
            rows,
            f"=== {video_id}  (GT rel {meta['gt_relations']}, pred rel {meta['pred_relations']} | "
            f"GT obj {meta['gt_objects']}, pred obj {meta['pred_objects']}){note} ===",
        )
        if not meta["lenient_available"]:
            missing_lenient.append(video_id)

    if not all_results:
        raise ValueError("No runnable prediction directories found.")

    # Macro average (mean over videos) per criterion/threshold.
    aggregate = {}
    for video_id, payload in all_results.items():
        for row in payload["rows"]:
            key = (row["criterion"], row["tiou"])
            aggregate.setdefault(key, []).append(row)

    agg_rows = []
    for (criterion, tiou), rows in sorted(aggregate.items()):
        def mean(field):
            vals = [r[field] for r in rows if r[field] is not None]
            return sum(vals) / len(vals) if vals else None

        agg_rows.append({
            "criterion": criterion,
            "tiou": tiou,
            "videos": len(rows),
            "triplet_recall": mean("triplet_recall"),
            "relation_recall": mean("relation_recall"),
            "object_accuracy": mean("object_accuracy"),
            "object_recall": mean("object_recall"),
            "attribute_recall": mean("attribute_recall"),
        })

    def fmt(value):
        return "n/a" if value is None else f"{value * 100:6.1f}%"

    print_table_style = f"{'criterion':<10} {'tIoU':>5} {'videos':>6} {'triplet_R':>10} {'relation_R':>11} {'obj_acc':>8} {'obj_rec':>8} {'attr_R':>7}"
    print(f"\n=== AGGREGATE (macro-average over videos) ===")
    print(print_table_style)
    print("-" * len(print_table_style))
    for row in agg_rows:
        print(
            f"{row['criterion']:<10} {row['tiou']:>5.1f} {row['videos']:>6} "
            f"{fmt(row['triplet_recall']):>10} {fmt(row['relation_recall']):>11} "
            f"{fmt(row['object_accuracy']):>8} {fmt(row['object_recall']):>8} {fmt(row['attribute_recall']):>7}"
        )

    if missing_lenient:
        print(
            f"\nNote: lenient rows missing for {len(missing_lenient)} video(s) "
            f"({', '.join(missing_lenient)}). Generate semantic_alignment.json via "
            f"evaluate_scenegraph.py --pred_dir (one-time judge calls), then rerun this report."
        )

    output_json = args.output_json or str(pred_dir / "metrics_report.json")
    Path(output_json).write_text(
        json.dumps({"per_video": all_results, "aggregate": agg_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nFull report saved to: {output_json}")


if __name__ == "__main__":
    main()
