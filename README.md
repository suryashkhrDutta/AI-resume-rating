# AI Resume Rating

A small command-line tool that compares a folder of resumes against a job description and ranks the candidates.

An LLM (via [Groq](https://groq.com)) reads the job description and each resume and decides which skills and requirements are matched. **The LLM never produces the score.** Plain Python turns those judgments into a fixed 100-point rubric, so every point in a score can be traced back to a specific match or gap.

## Features

- Reads resumes in **PDF** and **DOCX** format
- Extracts structured data from the job description and each resume, validated with Pydantic
- Semantic skill matching (e.g. "OOP" can match "object-oriented programming"), mapped back to the job description's exact skill names
- Transparent scoring with a per-category breakdown
- Matching vs missing skills, strengths, resume quality issues, and job-specific improvement suggestions per candidate
- Ranked summary table in the terminal
- Optional **JSON** and **CSV** export
- A failure on one resume is reported and skipped; the rest still run

## How it works

```
Job description ──► LLM: extract requirements ──┐
                                                ├──► LLM: compare & list matches ──► Python: score ──► ranked results
Each resume ──► read PDF/DOCX ──► LLM: extract ─┘
```

1. The job description is parsed once into required skills, preferred skills, minimum experience, education requirements and responsibilities.
2. Each resume is converted to text and parsed into a structured profile (skills, experience, education, projects, certifications).
3. The LLM compares the two and returns matched and missing skills, whether the experience and education requirements are met, supporting evidence, strengths, quality issues and suggestions. It is explicitly told not to give any score.
4. Python maps the LLM's matches back to the job's exact skill labels, drops anything that is not in the job description, and computes the score.

## Scoring rubric

| Category | Points | How it is calculated |
|---|---|---|
| Required skills | 50 | matched ÷ total required × 50 |
| Preferred skills | 15 | matched ÷ total preferred × 15 |
| Experience | 15 | Full points if the minimum experience is met, or if the job states none |
| Education | 10 | Full points if the education requirement is satisfied, or if the job states none |
| Relevant experience | 10 | Full points if the resume shows any relevant experience |

If the job description lists no skills in a category, that category is awarded in full.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A [Groq API key](https://console.groq.com/keys)

## Setup

```bash
git clone https://github.com/suryashkhrDutta/AI-resume-rating.git
cd AI-resume-rating
uv sync --no-install-project
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

## Usage

1. Put the resumes (`.pdf` or `.docx`) in the `resumes/` folder.
2. Save the job description as a text file, for example `job.txt`.
3. Run:

```bash
uv run --no-sync python -c "import sys; sys.path.insert(0, 'src'); from resumeRating import main; main()" --job-description-file job.txt
```

### Options

| Option | Description |
|---|---|
| `--job-description-file PATH` | Text file with the job description. If omitted, you are prompted to paste it (finish with Ctrl+Z then Enter on Windows, Ctrl+D on macOS/Linux) |
| `--resumes-dir PATH` | Folder containing the resumes (default: `resumes/` in the project root) |
| `--json-output PATH` | Save full results as JSON |
| `--csv-output PATH` | Save results as CSV |

Example with exports:

```bash
uv run --no-sync python -c "import sys; sys.path.insert(0, 'src'); from resumeRating import main; main()" \
  --job-description-file job.txt \
  --json-output output/results.json \
  --csv-output output/results.csv
```

## Output

The tool prints a ranked table, followed by a detailed section for each candidate. The layout looks like this (values are illustrative):

```
Candidate      Score  Required Skills  Experience
--------------------------------------------------
Candidate A    80.00       3/4              Yes
Candidate B    62.50       2/4              No

Candidate A | Overall Score: 80.00/100
  Required Skills:       37.50/50
  Preferred Skills:       7.50/15
  Experience:            15.00/15
  Education:             10.00/10
  Relevant Experience:   10.00/10
  Matching required: Python, Data Structures, OOP
  Missing required:  Git
  ...
  Strengths: ...
  Verdict: ...
  Quality issues: ...
  Improvement suggestions: ...
```

## Project structure

```
AI-resume-rating/
├── src/resumeRating/__init__.py   # models, LLM calls, scoring, file reading, CLI
├── resumes/                       # input resumes (PDF / DOCX)
├── pyproject.toml
└── uv.lock
```

## Limitations

- **Scores depend on the LLM's judgments.** The scoring is deterministic, but which skills count as matched comes from the model, so results can vary slightly between runs.
- Experience, education and relevant experience are all-or-nothing, so most of the score variation comes from skills.
- Scanned or image-only PDFs are not supported (there is no OCR); the resume must contain selectable text.
- Resume text is sent to the Groq API for processing. Do not use it with resumes you are not permitted to share.
- This is a small project and a decision-support aid, not a hiring decision maker. It has no automated tests yet.

## Tech stack

Python · [Groq API](https://groq.com) (`openai/gpt-oss-120b`) · Pydantic · pypdf · python-docx · python-dotenv · uv
