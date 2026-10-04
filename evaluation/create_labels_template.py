"""Generate a labels.json template for manual ground-truth labeling.

Reads the evaluation job description, extracts required skills using the
existing application logic (LLM call), scans for resume files, and produces
a labels.json file where each resume has:
  - required_skills: extracted from the JD (read-only reference)
  - ground_truth_matches: empty list for the user to manually fill

If labels.json already exists, existing ground_truth_matches entries are
preserved and only new resumes are added.
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project's src/ importable so we can reuse the production code.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from groq import Groq

from resumeRating import parse_job_description

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
JD_PATH = DATASET_DIR / "job_description.txt"
RESUMES_DIR = DATASET_DIR / "resumes"
LABELS_PATH = DATASET_DIR / "labels.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _discover_resumes(resumes_dir: Path) -> list[str]:
    """Return sorted filenames of supported resume files."""
    return sorted(
        f.name
        for f in resumes_dir.iterdir()
        if f.is_file() and f.suffix.casefold() in SUPPORTED_EXTENSIONS
    )


def _load_existing_labels(path: Path) -> dict:
    """Load existing labels.json if present, else return empty dict."""
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not read existing {path.name}: {exc}")
    return {}


def main() -> None:
    # ---- Validate dataset --------------------------------------------------
    if not JD_PATH.is_file():
        print(f"Error: Job description not found at {JD_PATH}")
        print("Place your job description text in that file and rerun.")
        sys.exit(1)

    jd_text = JD_PATH.read_text(encoding="utf-8").strip()
    if not jd_text or jd_text.startswith("Paste your job description here"):
        print("Error: job_description.txt still contains the placeholder text.")
        print("Replace it with an actual job description and rerun.")
        sys.exit(1)

    if not RESUMES_DIR.is_dir():
        print(f"Error: Resumes directory not found at {RESUMES_DIR}")
        sys.exit(1)

    resume_files = _discover_resumes(RESUMES_DIR)
    if not resume_files:
        print(f"Error: No PDF/DOCX resumes found in {RESUMES_DIR}")
        sys.exit(1)

    # ---- Parse JD to extract required skills --------------------------------
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    print(f"Parsing job description to extract required skills...")
    job = parse_job_description(jd_text, client)

    required_skills = job.required_skills
    print(f"Extracted {len(required_skills)} required skills: {required_skills}")

    # ---- Build / update labels.json ----------------------------------------
    existing = _load_existing_labels(LABELS_PATH)
    labels: dict = {}

    for filename in resume_files:
        if filename in existing:
            # Preserve existing ground_truth_matches
            entry = existing[filename]
            labels[filename] = {
                "required_skills": required_skills,
                "ground_truth_matches": entry.get("ground_truth_matches", []),
            }
            status = "preserved"
        else:
            labels[filename] = {
                "required_skills": required_skills,
                "ground_truth_matches": [],
            }
            status = "NEW"
        print(f"  {filename}: {status}")

    # Warn about resumes removed from the directory
    removed = set(existing.keys()) - set(resume_files)
    for r in sorted(removed):
        print(f"  {r}: file no longer in resumes/ — entry dropped")

    LABELS_PATH.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nLabels template saved to: {LABELS_PATH}")
    print()
    print("NEXT STEP:")
    print("  Open labels.json and fill 'ground_truth_matches' for each resume.")
    print("  List only the required skills that are genuinely present in that resume.")
    print("  Use the exact skill names from the 'required_skills' array.")
    print()
    print("  Example:")
    print('    "ground_truth_matches": ["Python", "SQL"]')
    print()
    print("  Then run: python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
