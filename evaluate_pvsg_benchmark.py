#!/usr/bin/env python3
"""
Parallel Multi-Key Benchmark Runner for PVSG Validation (VidOR, Epic Kitchens, Ego4D)
Distributes videos across multiple parallel workers, each using its own API key.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel Multi-Key Benchmark Runner for PVSG Validation Splits"
    )
    p.add_argument(
        "--dataset_root",
        default="./svg2_dataset",
        help="Root directory containing vidor/, epic_kitchen/, ego4d/.",
    )
    p.add_argument(
        "--gt_file",
        default="./svg2_dataset/PVSG dataset.json",
        help="Path to ground truth PVSG dataset JSON.",
    )
    p.add_argument(
        "--datasets",
        default="vidor,epic_kitchen,ego4d",
        help="Comma-separated datasets to evaluate (vidor,epic_kitchen,ego4d).",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "train", "all"],
        help="Dataset split to evaluate (default: val).",
    )
    p.add_argument(
        "--output_dir",
        default="./output/pvsg_val_benchmark",
        help="Output directory.",
    )
    p.add_argument(
        "--model_name",
        default="openai/moonshotai/kimi-k3",
        help="VLM agent model.",
    )
    p.add_argument(
        "--judge_model_name",
        default="openai/deepseek-ai/deepseek-v4.1-flash",
        help="Semantic judge model.",
    )
    p.add_argument(
        "--api_base",
        default="https://integrate.api.nvidia.com/v1",
        help="API base URL for agent.",
    )
    p.add_argument(
        "--judge_api_base",
        default="https://integrate.api.nvidia.com/v1",
        help="API base URL for judge.",
    )
    p.add_argument(
        "--api_keys",
        default=None,
        help="Comma-separated list of API keys (or env NVIDIA_API_KEYS / NVIDIA_API_KEY).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to number of API keys).",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Max steps per video.",
    )
    p.add_argument(
        "--lenient_semantic",
        action="store_true",
        default=True,
        help="Enable lenient semantic matching with judge.",
    )
    p.add_argument(
        "--no_lenient_semantic",
        action="store_false",
        dest="lenient_semantic",
        help="Disable lenient semantic matching with judge.",
    )
    p.add_argument(
        "--tiou_thresholds",
        default="0.1,0.3,0.5",
        help="Comma-separated tIoU thresholds.",
    )
    p.add_argument(
        "--limit_per_dataset",
        type=int,
        default=None,
        help="Optional limit of videos per dataset (for testing).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from evaluate_scenegraph.py.",
    )
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Force re-running already evaluated videos.",
    )
    return p.parse_args()


def load_api_keys(cli_keys: str = None) -> list[str]:
    raw_keys = (
        cli_keys
        or os.environ.get("NVIDIA_NIM_API")
        or os.environ.get("NVIDIA_API_KEYS")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if raw_keys:
        keys = [
            k.strip()
            for k in raw_keys.replace("\n", ",").replace(";", ",").split(",")
            if k.strip()
        ]
        if keys:
            return keys
    raise ValueError(
        "No API keys found! Set NVIDIA_NIM_API='key1,key2,...' in your environment or pass --api_keys."
    )


def find_dataset_videos(
    dataset_root: Path, gt_data: dict, dataset_name: str, split: str
) -> list[tuple[str, Path]]:
    split_dict = gt_data.get("split", {}).get(dataset_name, {})
    val_ids = set(split_dict.get(split, []))
    vdir = dataset_root / dataset_name / "videos"
    if not vdir.is_dir():
        print(f"Warning: Directory not found: {vdir}")
        return []

    # Case-insensitive match for .mp4, .MP4, etc.
    all_videos = [p for p in vdir.iterdir() if p.suffix.lower() == ".mp4"]
    matched = [(dataset_name, p) for p in all_videos if p.stem in val_ids]
    return sorted(matched, key=lambda x: x[1].name)


def run_video_job(job_args):
    (
        dataset_name,
        video_path,
        video_idx,
        total_videos,
        api_key,
        key_idx,
        args_dict,
    ) = job_args
    output_dir = Path(args_dict["output_dir"]) / dataset_name
    video_id = video_path.stem
    run_dir = output_dir / video_id
    eval_file = run_dir / "evaluation.json"

    # Resume check
    if not args_dict["no_resume"] and eval_file.is_file():
        try:
            eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
            if eval_data.get("object_strict"):
                print(
                    f"[Worker Key {key_idx+1}] [{dataset_name} {video_idx}/{total_videos}] SKIPPED (already evaluated): {video_path.name}"
                )
                return {
                    "dataset": dataset_name,
                    "video_id": video_id,
                    "status": "CACHED",
                    "evaluation": eval_data,
                    "elapsed": 0.0,
                    "error": None,
                }
        except Exception:
            pass

    key_mask = f"...{api_key[-4:]}" if len(api_key) > 4 else "key"
    print(
        f"[Worker Key {key_idx+1} ({key_mask})] [{dataset_name} {video_idx}/{total_videos}] Starting: {video_path.name}"
    )
    t0 = time.time()

    cmd = [
        sys.executable,
        "-u",
        "evaluate_scenegraph.py",
        "--video_path",
        str(video_path),
        "--gt_file",
        args_dict["gt_file"],
        "--dataset_name",
        dataset_name,
        "--output_dir",
        str(output_dir),
        "--model_name",
        args_dict["model_name"],
        "--judge_model_name",
        args_dict["judge_model_name"],
        "--api_base",
        args_dict["api_base"],
        "--judge_api_base",
        args_dict["judge_api_base"],
        "--api_key",
        api_key,
        "--judge_api_key",
        api_key,
        "--max_steps",
        str(args_dict["max_steps"]),
        "--tiou_thresholds",
        args_dict["tiou_thresholds"],
    ]
    if args_dict.get("lenient_semantic"):
        cmd.append("--lenient_semantic")
    if args_dict.get("verbose"):
        cmd.append("--verbose")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "eval_run.log"

    try:
        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                if line.startswith("[STEP ") and "ACTIONS" in line:
                    step_num = line.split("]")[0].replace("[STEP", "").split("/")[0].strip()
                    print(
                        f"[Worker Key {key_idx+1}] [{dataset_name} {video_idx}/{total_videos}] {video_id} -> Step {step_num}/10"
                    )
            proc.wait()
            retcode = proc.returncode

        elapsed = time.time() - t0

        if retcode != 0:
            err_msg = f"Process exited with code {retcode}. See {log_file}"
            print(
                f"[Worker Key {key_idx+1}] [{dataset_name} {video_idx}/{total_videos}] FAILED ({elapsed:.1f}s): {err_msg}"
            )
            return {
                "dataset": dataset_name,
                "video_id": video_id,
                "status": "FAILED",
                "evaluation": None,
                "elapsed": elapsed,
                "error": err_msg,
            }

        eval_data = None
        if eval_file.is_file():
            eval_data = json.loads(eval_file.read_text(encoding="utf-8"))

        r05 = 0.0
        if eval_data:
            r05 = eval_data.get("0.5", {}).get("recall", 0.0) * 100
        print(
            f"[Worker Key {key_idx+1}] [{dataset_name} {video_idx}/{total_videos}] DONE in {elapsed:.1f}s | Rel tIoU>=0.5 Recall: {r05:.1f}%"
        )
        return {
            "dataset": dataset_name,
            "video_id": video_id,
            "status": "SUCCESS",
            "evaluation": eval_data,
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(
            f"[Worker Key {key_idx+1}] [{dataset_name} {video_idx}/{total_videos}] EXCEPTION: {e}"
        )
        return {
            "dataset": dataset_name,
            "video_id": video_id,
            "status": "ERROR",
            "evaluation": None,
            "elapsed": elapsed,
            "error": str(e),
        }


def aggregate_dataset_metrics(results: list, tiou_thresholds: list) -> dict:
    successful = [r for r in results if r.get("evaluation")]
    if not successful:
        return {"count": 0}
    n = len(successful)
    res = {
        "count": n,
        "object_strict": {
            "precision": sum(
                r["evaluation"].get("object_strict", {}).get("precision", 0.0)
                for r in successful
            )
            / n,
            "recall": sum(
                r["evaluation"].get("object_strict", {}).get("recall", 0.0)
                for r in successful
            )
            / n,
        },
        "relations_strict": {},
        "object_lenient": {
            "precision": sum(
                r["evaluation"].get("object_lenient", {}).get("precision", 0.0)
                for r in successful
            )
            / n,
            "recall": sum(
                r["evaluation"].get("object_lenient", {}).get("recall", 0.0)
                for r in successful
            )
            / n,
        },
        "relations_lenient": {},
    }
    for t in tiou_thresholds:
        t_str = f"{t:.1f}" if isinstance(t, float) else str(t)
        res["relations_strict"][t_str] = {
            "precision": sum(
                (r["evaluation"].get(t_str) or r["evaluation"].get(str(t)) or {}).get("precision", 0.0)
                for r in successful
            )
            / n,
            "recall": sum(
                (r["evaluation"].get(t_str) or r["evaluation"].get(str(t)) or {}).get("recall", 0.0)
                for r in successful
            )
            / n,
        }
        res["relations_lenient"][t_str] = {
            "precision": sum(
                (
                    (r["evaluation"].get("lenient") or {}).get(t_str)
                    or (r["evaluation"].get("lenient") or {}).get(str(t))
                    or {}
                ).get("precision", 0.0)
                for r in successful
            )
            / n,
            "recall": sum(
                (
                    (r["evaluation"].get("lenient") or {}).get(t_str)
                    or (r["evaluation"].get("lenient") or {}).get(str(t))
                    or {}
                ).get("recall", 0.0)
                for r in successful
            )
            / n,
        }
    return res


def prevent_mac_sleep():
    """Prevents system and display sleep on macOS while this process runs."""
    if sys.platform == "darwin":
        try:
            pid = os.getpid()
            subprocess.Popen(
                ["caffeinate", "-d", "-i", "-m", "-u", "-w", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("[System] macOS caffeinate active: sleep is automatically disabled.")
        except Exception:
            pass


def main():
    prevent_mac_sleep()
    args = parse_args()
    api_keys = load_api_keys(args.api_keys)
    num_workers = args.num_workers or len(api_keys)
    print(
        f"Loaded {len(api_keys)} API key(s). Running with {num_workers} parallel worker(s)."
    )

    gt_path = Path(args.gt_file).expanduser().resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))

    target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    jobs = []
    video_plan = {}
    for ds in target_datasets:
        vids = find_dataset_videos(dataset_root, gt_data, ds, args.split)
        if args.limit_per_dataset:
            vids = vids[: args.limit_per_dataset]
        video_plan[ds] = vids
        print(f"Dataset '{ds}' ({args.split}): found {len(vids)} video(s)")

    total_all_videos = sum(len(vids) for vids in video_plan.values())
    print(
        f"\nTOTAL VIDEOS TO EVALUATE: {total_all_videos} across {len(target_datasets)} dataset(s)"
    )

    if total_all_videos == 0:
        print("No videos to evaluate. Exiting.")
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    job_index = 0
    all_job_specs = []
    for ds, vids in video_plan.items():
        for idx, (dname, vpath) in enumerate(vids, start=1):
            assigned_key_idx = job_index % len(api_keys)
            assigned_key = api_keys[assigned_key_idx]
            spec = (
                dname,
                vpath,
                idx,
                len(vids),
                assigned_key,
                assigned_key_idx,
                args_dict,
            )
            all_job_specs.append(spec)
            job_index += 1

    t_start = time.time()
    results = []
    print(f"\n========================================")
    print(f"LAUNCHING PARALLEL PVSG BENCHMARK")
    print(f"Agent Model: {args.model_name}")
    print(f"Judge Model: {args.judge_model_name}")
    print(f"Workers: {num_workers} | API Keys: {len(api_keys)}")
    print(f"========================================\n")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_video_job, spec): spec for spec in all_job_specs
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    total_elapsed = time.time() - t_start
    tiou_thresholds = [
        float(x.strip()) for x in args.tiou_thresholds.split(",") if x.strip()
    ]

    # Per dataset summaries
    dataset_summaries = {}
    for ds in target_datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        dataset_summaries[ds] = aggregate_dataset_metrics(
            ds_results, tiou_thresholds
        )

    # Global overall summary
    overall_summary = aggregate_dataset_metrics(results, tiou_thresholds)
    overall_summary["total_elapsed_seconds"] = total_elapsed
    overall_summary["datasets"] = dataset_summaries
    overall_summary["agent_model"] = args.model_name
    overall_summary["judge_model"] = args.judge_model_name

    summary_file = output_dir / "pvsg_benchmark_summary.json"
    summary_file.write_text(
        json.dumps(overall_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print Report
    print(f"\n========================================================")
    print(
        f"PVSG VALIDATION BENCHMARK RESULTS ({overall_summary.get('count', 0)}/{total_all_videos} completed in {total_elapsed/60:.1f}m)"
    )
    print(f"========================================================")
    for ds in target_datasets:
        d_metrics = dataset_summaries.get(ds, {})
        cnt = d_metrics.get("count", 0)
        if cnt == 0:
            print(f"\n[{ds.upper()}] No successful evaluations.")
            continue
        obj_p = d_metrics["object_strict"]["precision"] * 100
        obj_r = d_metrics["object_strict"]["recall"] * 100
        l_obj_r = d_metrics["object_lenient"]["recall"] * 100
        print(f"\n[{ds.upper()}] ({cnt} videos)")
        print(
            f"  Object Match: Strict Recall={obj_r:.2f}%, Lenient Recall={l_obj_r:.2f}% (Precision={obj_p:.2f}%)"
        )
        for t in tiou_thresholds:
            t_str = f"{t:.1f}" if isinstance(t, float) else str(t)
            s_r = d_metrics["relations_strict"].get(t_str, {}).get("recall", 0.0) * 100
            l_r = d_metrics["relations_lenient"].get(t_str, {}).get("recall", 0.0) * 100
            print(
                f"  Rel tIoU>={t}: Strict Recall={s_r:.2f}%, Lenient Recall={l_r:.2f}%"
            )

    print(f"\n[OVERALL MACRO PVSG BENCHMARK]")
    all_obj_r = overall_summary.get("object_strict", {}).get("recall", 0.0) * 100
    all_l_obj_r = overall_summary.get("object_lenient", {}).get("recall", 0.0) * 100
    print(
        f"  Object Match: Strict Recall={all_obj_r:.2f}%, Lenient Recall={all_l_obj_r:.2f}%"
    )
    for t in tiou_thresholds:
        t_str = f"{t:.1f}" if isinstance(t, float) else str(t)
        s_r = overall_summary.get("relations_strict", {}).get(t_str, {}).get("recall", 0.0) * 100
        l_r = overall_summary.get("relations_lenient", {}).get(t_str, {}).get("recall", 0.0) * 100
        print(
            f"  Rel tIoU>={t}: Strict Recall={s_r:.2f}%, Lenient Recall={l_r:.2f}%"
        )
    print(f"========================================================")
    print(f"Full benchmark results saved to: {summary_file}")


if __name__ == "__main__":
    main()
