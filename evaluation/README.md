# Evaluation Suite

Reproducible evaluation pipeline for the AI Resume Rating system. Measures
skill-matching accuracy, score consistency, ranking stability, and reliability
using real LLM calls against a manually labeled dataset.

## Quick Start

```bash
# 1. Place your job description
#    Edit evaluation/dataset/job_description.txt with the actual JD text.

# 2. Place resumes
#    Copy PDF/DOCX resumes into evaluation/dataset/resumes/

# 3. Generate the labels template
uv run python evaluation/create_labels_template.py

# 4. Manually edit labels.json (see "How to Label" below)

# 5. Run the full evaluation
uv run python evaluation/run_evaluation.py
```

## Directory Structure

```
evaluation/
├── dataset/
│   ├── job_description.txt      # The job description to evaluate against
│   ├── resumes/                 # PDF/DOCX resume files
│   └── labels.json              # Ground-truth labels (generated then manually edited)
├── results/
│   ├── evaluation_results.json  # Raw machine-readable results
│   └── evaluation_report.md     # Human-readable report
├── create_labels_template.py    # Generates labels.json from the JD
├── run_evaluation.py            # Runs the full evaluation suite
└── README.md                    # This file
```

## Step-by-Step Guide

### 1. Create the Dataset

#### Job Description

Save the full text of the job description to:

```
evaluation/dataset/job_description.txt
```

This must be the same JD you want to evaluate resumes against.

#### Resumes

Place resume files (`.pdf` or `.docx`) in:

```
evaluation/dataset/resumes/
```

You can add or remove resumes at any time. The evaluation code dynamically
discovers all supported files in this directory.

### 2. Generate Labels Template

```bash
uv run python evaluation/create_labels_template.py
```

This:
- Parses the JD using the existing LLM-based parser
- Extracts the required skills
- Creates `labels.json` with an entry per resume
- If `labels.json` already exists, preserves your existing labels and adds new resumes

### 3. Manually Label Ground Truth

Open `evaluation/dataset/labels.json` and fill `ground_truth_matches` for each resume.

#### How to Label

For each resume:

1. **Read the actual resume** (open the PDF/DOCX).
2. Look at the `required_skills` array — these are the skills extracted from the JD.
3. For each required skill, decide whether the resume **genuinely demonstrates** that skill.
4. Add matching skill names to `ground_truth_matches`, using the **exact** names from `required_skills`.

Example:

```json
{
  "candidate_resume.pdf": {
    "required_skills": ["Python", "FastAPI", "SQL", "Docker", "Git"],
    "ground_truth_matches": ["Python", "SQL", "Git"]
  }
}
```

**Rules:**
- Only include skills the resume actually demonstrates — do not guess.
- Use semantic judgment: if the resume says "PostgreSQL" and the required skill is "SQL", that counts.
- Use the exact skill name string from `required_skills`, not your own variation.
- If a resume has none of the required skills, leave the array empty: `[]`.
- Do **not** copy the system's predictions as ground truth. The whole point is independent human judgment.

### 4. Run the Evaluation

```bash
uv run python evaluation/run_evaluation.py
```

#### Options

| Flag | Description |
|------|-------------|
| `--num-runs N` | Number of repeated pipeline runs per resume (default: 5) |
| `--skip-baseline` | Skip the optional LLM-baseline experiment |

Example:

```bash
uv run python evaluation/run_evaluation.py --num-runs 3 --skip-baseline
```

### 5. Review Results

Results are saved in `evaluation/results/`:

- **`evaluation_results.json`** — Complete raw data including per-run scores, attempt logs, and all metrics.
- **`evaluation_report.md`** — Human-readable report with tables and measured numbers.

## What Each Metric Means

### Skill Matching

| Metric | Formula | Meaning |
|--------|---------|---------|
| True Positive (TP) | System says match AND human says match | Correct match |
| False Positive (FP) | System says match BUT human says no match | System over-claims |
| False Negative (FN) | System says no match BUT human says match | System misses a skill |
| Precision | TP / (TP + FP) | Of claimed matches, how many are correct? |
| Recall | TP / (TP + FN) | Of actual matches, how many did the system find? |
| F1 | 2 × P × R / (P + R) | Harmonic mean of precision and recall |

**Micro-averaged** means all TP/FP/FN counts are pooled across all resumes before computing the ratios.

### Score Consistency

| Metric | Meaning |
|--------|---------|
| Score range | max − min score across repeated runs for one resume |
| Std dev | Standard deviation of scores across runs |
| Max abs deviation | Largest distance of any single run from the mean |
| Identical scores | How many resumes got the exact same score every run |

The Python scoring function is deterministic. Variation comes from the LLM
extraction/matching step — different runs may extract slightly different skill
matches, leading to different scores.

### Ranking Stability

| Metric | Meaning |
|--------|---------|
| Identical ranking % | Percentage of runs where the resume ranking order matched run 1 |
| Spearman ρ | Rank correlation coefficient (1.0 = perfect agreement, requires ≥3 resumes) |

### Reliability

| Metric | Meaning |
|--------|---------|
| Success rate | Fraction of pipeline attempts that completed without error |
| Avg latency | Mean processing time per successful resume |
| P50 latency | Median processing time |
| P95 latency | 95th percentile — indicates worst-case typical performance |
| Total batch runtime | Wall-clock time for the entire evaluation run |

### Baseline Comparison

Compares score variation between:
- **LLM baseline**: The LLM directly generates a 0–100 score (no structure, no rubric)
- **Deterministic rubric**: The current system where Python calculates scores from structured LLM judgments

Lower score variation = more consistent = more reliable for ranking candidates.

## How to Interpret Results

### What the numbers tell you

- **High precision, lower recall**: The system is conservative — when it says a skill matches, it's usually right, but it misses some actual matches.
- **High recall, lower precision**: The system is aggressive — it finds most matches but also claims some false ones.
- **Low score range / std dev**: The system produces consistent scores across runs.
- **High ranking stability**: Candidate ordering is reliable.
- **Lower baseline variation vs rubric variation**: Would indicate the LLM is already consistent on its own (unlikely for free-form scoring).

### Which numbers are safe to use on a resume

✅ **Safe to cite** (after expanding dataset to 20+ resumes):
- Skill-matching precision, recall, and F1 (these are grounded in human labels)
- Score consistency metrics (objectively measured)
- Success rate and latency (objectively measured)
- Ranking stability with Spearman ρ (objectively measured)

⚠️ **Cite with caveats**:
- Results from < 10 resumes (state the sample size)
- Baseline comparison (state it was created for evaluation, not a production predecessor)

❌ **Do not cite**:
- Metrics from unfilled labels (skill matching is meaningless without ground truth)
- Numbers you haven't personally verified

## Adding More Resumes

1. Place new resumes in `evaluation/dataset/resumes/`
2. Re-run `create_labels_template.py` — it will add new entries while preserving existing labels
3. Fill `ground_truth_matches` for the new resumes
4. Re-run `run_evaluation.py`

The evaluation code discovers resumes dynamically. No code changes needed.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not set` | Add it to `.env` in the project root |
| Rate limiting errors | Wait a minute and retry, or reduce `--num-runs` |
| `labels.json` not found | Run `create_labels_template.py` first |
| Placeholder JD error | Replace the placeholder text in `job_description.txt` |
| Import errors | Run from the project root: `uv run python evaluation/run_evaluation.py` |
