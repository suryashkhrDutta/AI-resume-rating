"""Local HTTP interface for the existing AI Resume Rating workflow."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq

from resumeRating import parse_job_description, process_resume


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SAMPLE_RESUMES_DIRECTORY = Path(__file__).resolve().parent / "resumes"

app = FastAPI(title="AI Resume Rating API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _sample_resume_path(filename: str) -> Path:
    candidate = (SAMPLE_RESUMES_DIRECTORY / filename).resolve()
    if candidate.parent != SAMPLE_RESUMES_DIRECTORY.resolve() or not candidate.is_file():
        raise ValueError("The selected sample resume was not found.")
    if candidate.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError("The selected sample is not a supported resume format.")
    return candidate


def _resume_error_message(error: Exception) -> str:
    message = str(error)
    if "rate_limit" in message.casefold() or "rate limit" in message.casefold():
        return "The Groq rate limit was reached. Try this resume again in a moment."
    return message


@app.get("/api/samples")
def list_sample_resumes() -> dict[str, list[dict[str, int | str]]]:
    if not SAMPLE_RESUMES_DIRECTORY.is_dir():
        return {"resumes": []}
    return {
        "resumes": [
            {"name": path.name, "size": path.stat().st_size}
            for path in sorted(SAMPLE_RESUMES_DIRECTORY.iterdir())
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ]
    }


@app.get("/api/samples/{filename}")
def view_sample_resume(filename: str) -> FileResponse:
    try:
        path = _sample_resume_path(filename)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    media_type = "application/pdf" if path.suffix.casefold() == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type="inline")


@app.post("/api/analyze")
async def analyze_resumes(
    job_description: Annotated[str, Form(...)],
    resumes: Annotated[list[UploadFile] | None, File()] = None,
    selected_samples: Annotated[list[str] | None, Form()] = None,
) -> dict:
    job_text = job_description.strip()
    if not job_text:
        raise HTTPException(status_code=422, detail="Enter or upload a job description.")
    if not resumes and not selected_samples:
        raise HTTPException(status_code=422, detail="Upload at least one PDF or DOCX resume.")

    resume_paths: list[tuple[str, Path]] = []
    uploads: list[tuple[str, str, bytes]] = []
    failed_resumes: list[dict[str, str]] = []
    for filename in selected_samples or []:
        try:
            resume_paths.append((filename, _sample_resume_path(filename)))
        except ValueError as error:
            failed_resumes.append({"source_file": filename, "error": str(error)})

    for index, upload in enumerate(resumes or [], start=1):
        source_file = upload.filename or f"resume-{index}"
        extension = Path(source_file).suffix.casefold()
        contents = await upload.read()

        if extension not in SUPPORTED_EXTENSIONS:
            failed_resumes.append({"source_file": source_file, "error": "Only PDF and DOCX files are supported."})
        elif not contents:
            failed_resumes.append({"source_file": source_file, "error": "The uploaded file is empty."})
        elif len(contents) > MAX_UPLOAD_BYTES:
            failed_resumes.append({"source_file": source_file, "error": "The file exceeds the 10 MB limit."})
        else:
            uploads.append((source_file, extension, contents))

    if not resume_paths and not uploads:
        raise HTTPException(status_code=422, detail="No valid PDF or DOCX resumes were uploaded.")

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured for the local API.")

    try:
        client = Groq(api_key=api_key)
        job = parse_job_description(job_text, client)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"The job description could not be analyzed: {error}") from error

    candidates: list[dict] = []
    with TemporaryDirectory() as directory:
        temporary_directory = Path(directory)
        for index, (source_file, extension, contents) in enumerate(uploads):
            temporary_file = temporary_directory / f"resume-{index}{extension}"
            temporary_file.write_bytes(contents)
            resume_paths.append((source_file, temporary_file))

        for source_file, resume_path in resume_paths:
            try:
                result = process_resume(resume_path, job, client)
                candidates.append({**result.model_dump(), "source_file": source_file})
            except Exception as error:
                failed_resumes.append({"source_file": source_file, "error": _resume_error_message(error)})

    candidates.sort(key=lambda candidate: candidate["final_score"], reverse=True)
    return {
        "candidates": candidates,
        "failed_resumes": failed_resumes,
        "job": job.model_dump(),
        "score_limits": {
            "required_skills": 50,
            "preferred_skills": 15,
            "experience": 15,
            "education": 10,
            "relevant_experience": 10,
        },
    }
