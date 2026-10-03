const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function getSampleResumes() {
  let response;
  try {
    response = await fetch(`${API_URL}/api/samples`);
  } catch {
    throw new Error('Could not load the repository sample resumes.');
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload || !Array.isArray(payload.resumes)) {
    throw new Error('Could not load the repository sample resumes.');
  }
  return payload.resumes;
}

export function getSampleResumeUrl(filename) {
  return `${API_URL}/api/samples/${encodeURIComponent(filename)}`;
}

export async function analyzeResumes(jobDescription, sampleResumes, resumes) {
  const body = new FormData();
  body.append('job_description', jobDescription);
  sampleResumes.forEach((resume) => body.append('selected_samples', resume.name));
  resumes.forEach((resume) => body.append('resumes', resume));

  let response;
  try {
    response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body });
  } catch {
    throw new Error('Could not reach the local Python API. Start it and try again.');
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(payload?.detail || 'The analysis could not be completed.');
  if (!payload || !Array.isArray(payload.candidates) || !Array.isArray(payload.failed_resumes)) {
    throw new Error('The API returned an unexpected response.');
  }
  return payload;
}
