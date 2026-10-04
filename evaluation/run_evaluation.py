"""Reproducible evaluation suite for AI Resume Rating.

Runs five evaluation sections against the dataset in evaluation/dataset/:
  1. Skill matching evaluation (requires manual ground-truth labels)
  2. Score consistency (repeated pipeline runs)
  3. Ranking stability
  4. Reliability / batch processing metrics
  5. Optional LLM-generated-score baseline comparison

All results are saved to evaluation/results/ as JSON and Markdown.
"""

import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make the project's src/ importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from groq import Groq

from resumeRating import (
    MODEL,
    JobDescription,
    CandidateResult,
    _request_json,
    parse_job_description,
    process_resume,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
JD_PATH = DATASET_DIR / "job_description.txt"
RESUMES_DIR = DATASET_DIR / "resumes"
LABELS_PATH = DATASET_DIR / "labels.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
DEFAULT_NUM_RUNS = 5


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when the denominator is zero."""
    return numerator / denominator if denominator else 0.0


def _discover_resumes(resumes_dir: Path) -> list[Path]:
    return sorted(
        f for f in resumes_dir.iterdir()
        if f.is_file() and f.suffix.casefold() in SUPPORTED_EXTENSIONS
    )


def _percentile(data: list[float], p: float) -> float:
    """Simple nearest-rank percentile for a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (p / 100) * (len(sorted_data) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _spearman_rank_correlation(ranks_a: list[int], ranks_b: list[int]) -> float | None:
    """Spearman's rank correlation coefficient. Returns None if n < 3."""
    n = len(ranks_a)
    if n < 3:
        return None
    d_squared = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - (6 * d_squared) / (n * (n ** 2 - 1))


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Skill Matching Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_skill_matching(
    labels: dict,
    run_results: dict[str, list[CandidateResult]],
) -> dict:
    """Compare predicted required-skill matches against ground truth.

    Uses the first successful run's predictions for each resume.
    Returns per-resume and micro-averaged metrics.
    """
    per_resume: list[dict] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    has_any_labels = False

    for filename, label_entry in labels.items():
        gt_matches = set(label_entry.get("ground_truth_matches", []))
        required_skills = set(label_entry.get("required_skills", []))

        if not gt_matches and not required_skills:
            continue  # skip if no labels at all

        # Check if user has actually filled in labels
        # (ground_truth_matches is non-empty OR the user explicitly set it empty
        #  meaning the resume has none of the required skills)
        # We need a way to distinguish "not yet labeled" from "labeled as empty".
        # Convention: if ground_truth_matches key exists and is a list, it's labeled.
        # If labels were generated but never edited, all will be empty lists.
        # We'll check below and warn.

        if not gt_matches:
            # Could be "not yet labeled" or "genuinely has no matching skills"
            # We'll include it but track this.
            pass

        has_any_labels = has_any_labels or bool(gt_matches)

        # Get the system's predictions from the first successful run
        runs = run_results.get(filename, [])
        if not runs:
            per_resume.append({
                "resume": filename,
                "error": "No successful pipeline run — cannot evaluate skill matching",
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            })
            continue

        first_result = runs[0]
        predicted_matches = set(first_result.matching_required_skills)

        tp = len(predicted_matches & gt_matches)
        fp = len(predicted_matches - gt_matches)
        fn = len(gt_matches - predicted_matches)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_resume.append({
            "resume": filename,
            "required_skills": sorted(required_skills),
            "ground_truth_matches": sorted(gt_matches),
            "predicted_matches": sorted(predicted_matches),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "has_ground_truth_labels": has_any_labels,
        "warning": (
            None if has_any_labels
            else "No ground_truth_matches have been filled in. "
                 "Run create_labels_template.py, manually edit labels.json, then rerun."
        ),
        "micro_averaged": {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "per_resume": per_resume,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Score Consistency
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_score_consistency(
    run_results: dict[str, list[CandidateResult]],
    num_runs: int,
) -> dict:
    """Measure score variation across repeated runs."""
    per_resume: list[dict] = []
    all_ranges: list[float] = []
    all_stds: list[float] = []
    all_max_devs: list[float] = []
    identical_count = 0

    for filename, results in run_results.items():
        scores = [r.final_score for r in results]
        if not scores:
            per_resume.append({
                "resume": filename,
                "error": "No successful runs",
                "scores": [],
            })
            continue

        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        max_abs_dev = max(abs(s - mean_score) for s in scores)

        is_identical = score_range == 0.0
        if is_identical:
            identical_count += 1

        all_ranges.append(score_range)
        all_stds.append(std_dev)
        all_max_devs.append(max_abs_dev)

        per_resume.append({
            "resume": filename,
            "num_successful_runs": len(scores),
            "scores": scores,
            "min_score": min_score,
            "max_score": max_score,
            "score_range": score_range,
            "mean_score": mean_score,
            "std_dev": std_dev,
            "max_abs_deviation_from_mean": max_abs_dev,
            "identical_across_all_runs": is_identical,
        })

    total_resumes = len(run_results)
    return {
        "num_runs_per_resume": num_runs,
        "total_resumes": total_resumes,
        "resumes_with_identical_scores": identical_count,
        "aggregate": {
            "mean_score_range": statistics.mean(all_ranges) if all_ranges else None,
            "max_score_range": max(all_ranges) if all_ranges else None,
            "mean_std_dev": statistics.mean(all_stds) if all_stds else None,
            "max_std_dev": max(all_stds) if all_stds else None,
            "mean_max_abs_deviation": statistics.mean(all_max_devs) if all_max_devs else None,
            "max_max_abs_deviation": max(all_max_devs) if all_max_devs else None,
        },
        "per_resume": per_resume,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Ranking Stability
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_ranking_stability(
    run_results: dict[str, list[CandidateResult]],
    num_runs: int,
) -> dict:
    """Compare resume rankings across repeated runs."""
    filenames = sorted(run_results.keys())
    num_resumes = len(filenames)

    if num_resumes < 2:
        return {
            "note": "Ranking stability requires at least 2 resumes.",
            "num_resumes": num_resumes,
        }

    # Determine max number of successful runs across all resumes
    max_runs = min(len(run_results[f]) for f in filenames) if filenames else 0
    if max_runs < 2:
        return {
            "note": "Need at least 2 successful runs for ranking comparison.",
            "max_common_runs": max_runs,
        }

    # Build rankings per run
    rankings_per_run: list[list[str]] = []
    for run_idx in range(max_runs):
        scored = []
        for f in filenames:
            results = run_results[f]
            if run_idx < len(results):
                scored.append((f, results[run_idx].final_score))
        # Sort by score descending, then filename for tie-breaking
        scored.sort(key=lambda x: (-x[1], x[0]))
        rankings_per_run.append([s[0] for s in scored])

    reference_ranking = rankings_per_run[0]
    ref_rank_map = {name: rank for rank, name in enumerate(reference_ranking)}
    ref_ranks = [ref_rank_map[f] for f in filenames]

    comparisons: list[dict] = []
    identical_count = 0

    for run_idx in range(1, max_runs):
        run_ranking = rankings_per_run[run_idx]
        is_identical = run_ranking == reference_ranking

        if is_identical:
            identical_count += 1

        run_rank_map = {name: rank for rank, name in enumerate(run_ranking)}
        run_ranks = [run_rank_map[f] for f in filenames]
        spearman = _spearman_rank_correlation(ref_ranks, run_ranks)

        comparisons.append({
            "run": run_idx + 1,
            "vs_run": 1,
            "identical_ranking": is_identical,
            "spearman_rho": spearman,
            "ranking": run_ranking,
        })

    total_comparisons = max_runs - 1
    return {
        "num_resumes": num_resumes,
        "num_runs_compared": max_runs,
        "reference_ranking_run_1": reference_ranking,
        "identical_ranking_count": identical_count,
        "total_comparisons": total_comparisons,
        "identical_ranking_percentage": _safe_div(identical_count, total_comparisons) * 100,
        "comparisons": comparisons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Reliability / Batch Processing
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate_reliability(attempt_logs: list[dict]) -> dict:
    """Calculate reliability metrics from attempt logs."""
    total = len(attempt_logs)
    successes = [a for a in attempt_logs if a["success"]]
    failures = [a for a in attempt_logs if not a["success"]]
    latencies = [a["elapsed_seconds"] for a in successes]

    return {
        "total_attempts": total,
        "successful": len(successes),
        "failed": len(failures),
        "success_rate": _safe_div(len(successes), total),
        "latency": {
            "mean_seconds": statistics.mean(latencies) if latencies else None,
            "median_p50_seconds": statistics.median(latencies) if latencies else None,
            "p95_seconds": _percentile(latencies, 95) if latencies else None,
            "min_seconds": min(latencies) if latencies else None,
            "max_seconds": max(latencies) if latencies else None,
        },
        "failed_resumes": [
            {
                "resume": a["resume"],
                "run": a["run"],
                "exception_type": a.get("exception_type"),
                "exception_message": a.get("exception_message"),
            }
            for a in failures
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Optional LLM Baseline (isolated)
# ═══════════════════════════════════════════════════════════════════════════

def _llm_baseline_score(
    jd_text: str,
    resume_path: Path,
    client: Groq,
) -> float:
    """Ask the LLM to directly produce a 0-100 score. Isolated baseline."""
    from resumeRating import read_resume

    resume_text = read_resume(resume_path)
    system_prompt = (
        "You are an HR recruiter. Read the job description and the candidate's "
        "resume below. Return ONLY a JSON object with a single key 'score' whose "
        "value is an integer from 0 to 100 representing how well the candidate "
        "matches the job. Do not explain your reasoning."
    )
    user_prompt = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE RESUME:\n{resume_text}"
    )
    data = _request_json(client, system_prompt, user_prompt)
    score = data.get("score")
    if score is None:
        raise ValueError(f"LLM baseline did not return a 'score' key. Got: {data}")
    return float(score)


def _run_baseline_experiment(
    jd_text: str,
    resume_files: list[Path],
    client: Groq,
    num_runs: int,
) -> dict:
    """Run the LLM-baseline experiment: ask LLM to score directly, repeated."""
    per_resume: list[dict] = []
    all_ranges: list[float] = []
    all_stds: list[float] = []

    for resume_path in resume_files:
        filename = resume_path.name
        scores: list[float] = []
        errors: list[str] = []

        for run_idx in range(num_runs):
            try:
                score = _llm_baseline_score(jd_text, resume_path, client)
                scores.append(score)
            except Exception as exc:
                errors.append(f"Run {run_idx + 1}: {exc}")

        entry: dict[str, Any] = {"resume": filename, "scores": scores, "errors": errors}
        if scores:
            mean_s = statistics.mean(scores)
            std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
            entry["min_score"] = min(scores)
            entry["max_score"] = max(scores)
            entry["score_range"] = max(scores) - min(scores)
            entry["mean_score"] = mean_s
            entry["std_dev"] = std_s
            all_ranges.append(entry["score_range"])
            all_stds.append(std_s)

        per_resume.append(entry)

    return {
        "description": (
            "LLM-generated-score baseline: the LLM directly produces a 0-100 score. "
            "This is compared against the current architecture where Python calculates "
            "the final score deterministically from structured LLM judgments."
        ),
        "note": (
            "Baseline unavailable — no previous LLM-generated scoring implementation "
            "exists in the production codebase. This baseline was created solely for "
            "this evaluation experiment and is isolated under evaluation/."
        ),
        "num_runs_per_resume": num_runs,
        "aggregate": {
            "mean_score_range": statistics.mean(all_ranges) if all_ranges else None,
            "max_score_range": max(all_ranges) if all_ranges else None,
            "mean_std_dev": statistics.mean(all_stds) if all_stds else None,
            "max_std_dev": max(all_stds) if all_stds else None,
        },
        "per_resume": per_resume,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Resume Rating evaluation suite")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Number of repeated pipeline runs per resume (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the optional LLM-baseline experiment")
    args = parser.parse_args()
    num_runs = args.num_runs

    # ---- Validate environment ----------------------------------------------
    if not JD_PATH.is_file():
        print(f"Error: {JD_PATH} not found.")
        sys.exit(1)

    jd_text = JD_PATH.read_text(encoding="utf-8").strip()
    if not jd_text or jd_text.startswith("Paste your job description here"):
        print("Error: job_description.txt still contains the placeholder text.")
        sys.exit(1)

    if not RESUMES_DIR.is_dir():
        print(f"Error: {RESUMES_DIR} not found.")
        sys.exit(1)

    resume_files = _discover_resumes(RESUMES_DIR)
    if not resume_files:
        print("Error: No PDF/DOCX resumes found in the dataset.")
        sys.exit(1)

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    # ---- Parse JD once -----------------------------------------------------
    print(f"Parsing job description...")
    job = parse_job_description(jd_text, client)
    print(f"  Role: {job.role}")
    print(f"  Required skills ({len(job.required_skills)}): {job.required_skills}")
    print(f"  Preferred skills ({len(job.preferred_skills)}): {job.preferred_skills}")
    print()

    # ---- Run pipeline N times per resume -----------------------------------
    print(f"Running pipeline {num_runs} time(s) per resume ({len(resume_files)} resumes)...")
    print(f"Total pipeline calls: {num_runs * len(resume_files)}")
    print()

    # run_results: filename -> list of CandidateResult (one per successful run)
    run_results: dict[str, list[CandidateResult]] = OrderedDict()
    attempt_logs: list[dict] = []
    batch_start = time.time()

    for resume_path in resume_files:
        filename = resume_path.name
        run_results[filename] = []

        for run_idx in range(num_runs):
            run_label = f"[{filename}] run {run_idx + 1}/{num_runs}"
            print(f"  {run_label}...", end=" ", flush=True)

            start_time = time.time()
            attempt: dict[str, Any] = {
                "resume": filename,
                "run": run_idx + 1,
                "start_time_iso": datetime.now(timezone.utc).isoformat(),
            }

            try:
                result = process_resume(resume_path, job, client)
                elapsed = time.time() - start_time
                attempt["success"] = True
                attempt["elapsed_seconds"] = round(elapsed, 3)
                attempt["end_time_iso"] = datetime.now(timezone.utc).isoformat()
                attempt["score"] = result.final_score
                run_results[filename].append(result)
                print(f"score={result.final_score:.2f} ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.time() - start_time
                attempt["success"] = False
                attempt["elapsed_seconds"] = round(elapsed, 3)
                attempt["end_time_iso"] = datetime.now(timezone.utc).isoformat()
                attempt["exception_type"] = type(exc).__name__
                attempt["exception_message"] = str(exc)
                print(f"FAILED: {exc} ({elapsed:.1f}s)")

            attempt_logs.append(attempt)

    batch_elapsed = round(time.time() - batch_start, 3)
    print(f"\nPipeline runs complete. Total time: {batch_elapsed:.1f}s")
    print()

    # ---- Load labels for skill matching ------------------------------------
    labels: dict = {}
    if LABELS_PATH.is_file():
        try:
            labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Could not read labels.json: {exc}")

    # ---- Compute all metrics -----------------------------------------------
    skill_metrics = _evaluate_skill_matching(labels, run_results)
    consistency_metrics = _evaluate_score_consistency(run_results, num_runs)
    ranking_metrics = _evaluate_ranking_stability(run_results, num_runs)
    reliability_metrics = _aggregate_reliability(attempt_logs)
    reliability_metrics["total_batch_runtime_seconds"] = batch_elapsed

    # ---- Optional baseline experiment --------------------------------------
    baseline_metrics: dict | None = None
    if not args.skip_baseline:
        print("Running optional LLM-baseline experiment...")
        try:
            baseline_metrics = _run_baseline_experiment(
                jd_text, resume_files, client, num_runs
            )
            print("Baseline experiment complete.")
        except Exception as exc:
            print(f"Baseline experiment failed: {exc}")
            baseline_metrics = {"error": str(exc)}
    else:
        print("LLM-baseline experiment skipped (--skip-baseline).")

    print()

    # ---- Assemble full results ---------------------------------------------
    timestamp = datetime.now(timezone.utc).isoformat()
    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "model": MODEL,
            "num_resumes": len(resume_files),
            "num_runs_per_resume": num_runs,
            "jd_filename": JD_PATH.name,
            "resume_filenames": [f.name for f in resume_files],
        },
        "skill_matching": skill_metrics,
        "score_consistency": consistency_metrics,
        "ranking_stability": ranking_metrics,
        "reliability": reliability_metrics,
        "baseline_comparison": baseline_metrics,
        "raw_attempt_logs": attempt_logs,
    }

    # ---- Save results ------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_json_path = RESULTS_DIR / "evaluation_results.json"
    results_json_path.write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Raw results saved to: {results_json_path}")

    # ---- Generate human-readable report ------------------------------------
    report = _generate_report(full_results)
    report_path = RESULTS_DIR / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # ---- Print summary to console ------------------------------------------
    _print_summary(full_results)


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_report(results: dict) -> str:
    """Generate a human-readable Markdown evaluation report."""
    meta = results["metadata"]
    skill = results["skill_matching"]
    consistency = results["score_consistency"]
    ranking = results["ranking_stability"]
    reliability = results["reliability"]
    baseline = results.get("baseline_comparison")

    lines: list[str] = []
    w = lines.append  # shorthand

    w("# AI Resume Rating — Evaluation Report\n")
    w(f"**Generated**: {meta['timestamp']}  ")
    w(f"**Model**: `{meta['model']}`  ")
    w(f"**Resumes evaluated**: {meta['num_resumes']}  ")
    w(f"**Runs per resume**: {meta['num_runs_per_resume']}  ")
    w(f"**Job description**: `{meta['jd_filename']}`\n")

    # Dataset size caveat
    if meta["num_resumes"] < 10:
        w("> **⚠️ Small dataset warning**: This evaluation uses only "
          f"{meta['num_resumes']} resume(s). Results should be interpreted as "
          "preliminary indicators, not statistically robust claims. Expand the "
          "dataset to 20–30 resumes before citing these numbers on a resume.\n")

    w("---\n")

    # -- Section 1: Skill Matching -------------------------------------------
    w("## 1. Skill Matching Evaluation\n")
    if skill.get("warning"):
        w(f"> **⚠️ Warning**: {skill['warning']}\n")
        w("Skill matching metrics require manual ground-truth labels in `labels.json`.\n")
    else:
        micro = skill["micro_averaged"]
        w("### Micro-Averaged Metrics (across all skill decisions)\n")
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| True Positives | {micro['total_tp']} |")
        w(f"| False Positives | {micro['total_fp']} |")
        w(f"| False Negatives | {micro['total_fn']} |")
        w(f"| **Precision** | **{micro['precision']:.4f}** |")
        w(f"| **Recall** | **{micro['recall']:.4f}** |")
        w(f"| **F1 Score** | **{micro['f1']:.4f}** |")
        w("")

        w("### Per-Resume Metrics\n")
        w("| Resume | TP | FP | FN | Precision | Recall | F1 |")
        w("|--------|----|----|-----|-----------|--------|-----|")
        for entry in skill["per_resume"]:
            if "error" in entry:
                w(f"| {entry['resume']} | — | — | — | — | — | {entry['error']} |")
            else:
                w(f"| {entry['resume'][:40]} | {entry['tp']} | {entry['fp']} | "
                  f"{entry['fn']} | {entry['precision']:.4f} | "
                  f"{entry['recall']:.4f} | {entry['f1']:.4f} |")
        w("")

    w("---\n")

    # -- Section 2: Score Consistency ----------------------------------------
    w("## 2. Score Consistency\n")
    agg = consistency["aggregate"]
    w(f"**Runs per resume**: {consistency['num_runs_per_resume']}  ")
    w(f"**Total resumes**: {consistency['total_resumes']}  ")
    w(f"**Resumes with identical scores across all runs**: "
      f"{consistency['resumes_with_identical_scores']}/{consistency['total_resumes']}\n")

    w("### Aggregate Statistics\n")
    w("| Metric | Value |")
    w("|--------|-------|")
    _agg_row = lambda label, val: w(f"| {label} | {val:.4f} |" if val is not None else f"| {label} | N/A |")
    _agg_row("Mean score range", agg.get("mean_score_range"))
    _agg_row("Max score range", agg.get("max_score_range"))
    _agg_row("Mean std dev", agg.get("mean_std_dev"))
    _agg_row("Max std dev", agg.get("max_std_dev"))
    _agg_row("Mean max |deviation|", agg.get("mean_max_abs_deviation"))
    _agg_row("Max max |deviation|", agg.get("max_max_abs_deviation"))
    w("")

    w("### Per-Resume Scores\n")
    w("| Resume | Scores | Range | Mean | Std Dev | Identical? |")
    w("|--------|--------|-------|------|---------|------------|")
    for entry in consistency["per_resume"]:
        if "error" in entry:
            w(f"| {entry['resume'][:40]} | — | — | — | — | {entry['error']} |")
        else:
            scores_str = ", ".join(f"{s:.2f}" for s in entry["scores"])
            w(f"| {entry['resume'][:40]} | {scores_str} | "
              f"{entry['score_range']:.2f} | {entry['mean_score']:.2f} | "
              f"{entry['std_dev']:.4f} | "
              f"{'✅' if entry['identical_across_all_runs'] else '❌'} |")
    w("")

    w("---\n")

    # -- Section 3: Ranking Stability ----------------------------------------
    w("## 3. Ranking Stability\n")
    if "note" in ranking:
        w(f"> {ranking['note']}\n")
    else:
        w(f"**Runs compared**: {ranking['num_runs_compared']}  ")
        w(f"**Identical rankings**: {ranking['identical_ranking_count']}"
          f"/{ranking['total_comparisons']} "
          f"({ranking['identical_ranking_percentage']:.1f}%)\n")

        w("**Reference ranking (run 1)**:\n")
        for i, name in enumerate(ranking["reference_ranking_run_1"], 1):
            w(f"  {i}. {name}")
        w("")

        w("| Run | Identical? | Spearman ρ |")
        w("|-----|-----------|------------|")
        for comp in ranking["comparisons"]:
            rho = f"{comp['spearman_rho']:.4f}" if comp["spearman_rho"] is not None else "N/A"
            w(f"| Run {comp['run']} | "
              f"{'✅' if comp['identical_ranking'] else '❌'} | {rho} |")
        w("")

    w("---\n")

    # -- Section 4: Reliability ----------------------------------------------
    w("## 4. Reliability / Batch Processing\n")
    lat = reliability["latency"]
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Total attempts | {reliability['total_attempts']} |")
    w(f"| Successful | {reliability['successful']} |")
    w(f"| Failed | {reliability['failed']} |")
    w(f"| **Success rate** | **{reliability['success_rate']:.4f}** |")
    _lat_row = lambda label, val: w(f"| {label} | {val:.3f}s |" if val is not None else f"| {label} | N/A |")
    _lat_row("Avg latency", lat.get("mean_seconds"))
    _lat_row("Median (P50) latency", lat.get("median_p50_seconds"))
    _lat_row("P95 latency", lat.get("p95_seconds"))
    _lat_row("Min latency", lat.get("min_seconds"))
    _lat_row("Max latency", lat.get("max_seconds"))
    w(f"| Total batch runtime | {reliability['total_batch_runtime_seconds']:.3f}s |")
    w("")

    if reliability["failed_resumes"]:
        w("### Failures\n")
        w("| Resume | Run | Exception | Message |")
        w("|--------|-----|-----------|---------|")
        for f in reliability["failed_resumes"]:
            w(f"| {f['resume'][:30]} | {f['run']} | "
              f"`{f.get('exception_type', '?')}` | {f.get('exception_message', '')[:60]} |")
        w("")

    w("---\n")

    # -- Section 5: Baseline Comparison --------------------------------------
    w("## 5. Optional Baseline: LLM-Generated Score vs Deterministic Rubric\n")
    if baseline is None:
        w("Baseline experiment was skipped.\n")
    elif "error" in baseline:
        w(f"Baseline experiment failed: {baseline['error']}\n")
    else:
        w(f"> {baseline['note']}\n")
        w(f"**Description**: {baseline['description']}\n")

        bagg = baseline["aggregate"]
        w("### LLM Baseline Score Variation\n")
        w("| Metric | Value |")
        w("|--------|-------|")
        _agg_row("Mean score range", bagg.get("mean_score_range"))
        _agg_row("Max score range", bagg.get("max_score_range"))
        _agg_row("Mean std dev", bagg.get("mean_std_dev"))
        _agg_row("Max std dev", bagg.get("max_std_dev"))
        w("")

        w("### Per-Resume Baseline Scores\n")
        w("| Resume | Scores | Range | Mean | Std Dev |")
        w("|--------|--------|-------|------|---------|")
        for entry in baseline["per_resume"]:
            if entry.get("scores"):
                scores_str = ", ".join(f"{s:.1f}" for s in entry["scores"])
                w(f"| {entry['resume'][:40]} | {scores_str} | "
                  f"{entry.get('score_range', 0):.1f} | "
                  f"{entry.get('mean_score', 0):.1f} | "
                  f"{entry.get('std_dev', 0):.4f} |")
            else:
                w(f"| {entry['resume'][:40]} | — | — | — | — |")
        w("")

        # Side-by-side comparison
        w("### Variation Comparison: Deterministic Rubric vs LLM Baseline\n")
        det_agg = results["score_consistency"]["aggregate"]
        w("| Metric | Deterministic Rubric | LLM Baseline |")
        w("|--------|---------------------|--------------|")
        _cmp = lambda label, a, b: w(
            f"| {label} | "
            f"{a:.4f} | " if a is not None else f"| {label} | N/A | "
            f"{b:.4f} |" if b is not None else "N/A |"
        )
        # Do it more carefully to handle None
        for label, dkey, bkey in [
            ("Mean score range", "mean_score_range", "mean_score_range"),
            ("Max score range", "max_score_range", "max_score_range"),
            ("Mean std dev", "mean_std_dev", "mean_std_dev"),
            ("Max std dev", "max_std_dev", "max_std_dev"),
        ]:
            dv = det_agg.get(dkey)
            bv = bagg.get(bkey)
            dstr = f"{dv:.4f}" if dv is not None else "N/A"
            bstr = f"{bv:.4f}" if bv is not None else "N/A"
            w(f"| {label} | {dstr} | {bstr} |")
        w("")

    w("---\n")

    # -- Interpretation notes ------------------------------------------------
    w("## Interpretation Notes\n")
    w("- **Precision** = TP / (TP + FP): Of the skills the system says match, how many actually do?")
    w("- **Recall** = TP / (TP + FN): Of the skills that actually match, how many does the system find?")
    w("- **F1** = harmonic mean of precision and recall.")
    w("- **Score range** = max − min across repeated runs. Lower is more consistent.")
    w("- **Std dev** measures score spread. Zero means perfectly identical scores.")
    w("- **Spearman ρ** measures ranking agreement. 1.0 = identical ranking.")
    w("- **P95 latency** = 95th percentile of processing time per resume.\n")
    w("The Python scoring function is deterministic given fixed structured inputs. "
      "However, the LLM extraction/matching step introduces variation, so the "
      "end-to-end pipeline is NOT guaranteed to produce identical results across runs.\n")

    return "\n".join(lines) + "\n"


def _print_summary(results: dict) -> None:
    """Print a concise summary to the console."""
    meta = results["metadata"]
    skill = results["skill_matching"]
    consistency = results["score_consistency"]
    reliability = results["reliability"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model:   {meta['model']}")
    print(f"Resumes: {meta['num_resumes']}")
    print(f"Runs:    {meta['num_runs_per_resume']} per resume")
    print()

    # Skill matching
    if skill.get("has_ground_truth_labels"):
        micro = skill["micro_averaged"]
        print("SKILL MATCHING (micro-averaged):")
        print(f"  Precision: {micro['precision']:.4f}")
        print(f"  Recall:    {micro['recall']:.4f}")
        print(f"  F1:        {micro['f1']:.4f}")
    else:
        print("SKILL MATCHING: Awaiting manual ground-truth labels in labels.json")
    print()

    # Consistency
    agg = consistency["aggregate"]
    print("SCORE CONSISTENCY:")
    print(f"  Identical scores across all runs: "
          f"{consistency['resumes_with_identical_scores']}/{consistency['total_resumes']}")
    if agg.get("mean_score_range") is not None:
        print(f"  Mean score range: {agg['mean_score_range']:.4f}")
        print(f"  Max score range:  {agg['max_score_range']:.4f}")
        print(f"  Mean std dev:     {agg['mean_std_dev']:.4f}")
    print()

    # Ranking
    ranking = results["ranking_stability"]
    if "identical_ranking_percentage" in ranking:
        print("RANKING STABILITY:")
        print(f"  Identical rankings: {ranking['identical_ranking_count']}"
              f"/{ranking['total_comparisons']} "
              f"({ranking['identical_ranking_percentage']:.1f}%)")
    print()

    # Reliability
    lat = reliability["latency"]
    print("RELIABILITY:")
    print(f"  Success rate: {reliability['success_rate']:.4f} "
          f"({reliability['successful']}/{reliability['total_attempts']})")
    if lat.get("mean_seconds") is not None:
        print(f"  Avg latency:  {lat['mean_seconds']:.3f}s")
        print(f"  P50 latency:  {lat['median_p50_seconds']:.3f}s")
        print(f"  P95 latency:  {lat['p95_seconds']:.3f}s")
    print(f"  Total time:   {reliability['total_batch_runtime_seconds']:.1f}s")

    # Baseline
    baseline = results.get("baseline_comparison")
    if baseline and "aggregate" in baseline:
        bagg = baseline["aggregate"]
        print()
        print("BASELINE COMPARISON (LLM-direct vs deterministic rubric):")
        det_agg = consistency["aggregate"]
        det_r = det_agg.get("mean_score_range")
        bas_r = bagg.get("mean_score_range")
        if det_r is not None and bas_r is not None:
            print(f"  Deterministic rubric mean range: {det_r:.4f}")
            print(f"  LLM baseline mean range:         {bas_r:.4f}")

    print()
    if meta["num_resumes"] < 10:
        print("⚠️  Small dataset ({} resumes). Expand to 20–30 for stronger claims.".format(
            meta["num_resumes"]))

    print("=" * 60)


if __name__ == "__main__":
    main()
