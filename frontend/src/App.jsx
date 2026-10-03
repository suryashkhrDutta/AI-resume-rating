import { useEffect, useRef, useState } from 'react';
import { analyzeResumes, getSampleResumes } from './api';
import ScreeningForm from './components/ScreeningForm';
import ScreeningResults from './components/ScreeningResults';

const ALLOWED_RESUMES = new Set(['pdf', 'docx']);
const BREAKDOWN_LABELS = {
  required_skills: 'Required skills',
  preferred_skills: 'Preferred skills',
  experience: 'Experience',
  education: 'Education',
  relevant_experience: 'Relevant experience',
};

function extension(file) {
  return file.name.split('.').pop()?.toLowerCase() || '';
}

function fileSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function List({ items, empty }) {
  return items?.length ? <ul className="plain-list">{items.map((item) => <li key={item}>{item}</li>)}</ul> : <p className="muted">{empty}</p>;
}

function Tags({ items, empty }) {
  return items?.length ? <div className="tag-list">{items.map((item) => <span key={item}>{item}</span>)}</div> : <p className="muted">{empty}</p>;
}

function Status({ value, label }) {
  const text = value === null ? 'N/A' : value ? 'Yes' : 'No';
  return <li><strong className={value ? 'pass' : 'neutral'}>{text}</strong><span>{label}</span></li>;
}

function UploadForm({ onAnalyze, loading, apiError }) {
  const [jobDescription, setJobDescription] = useState('');
  const [jobFile, setJobFile] = useState('');
  const [resumes, setResumes] = useState([]);
  const [formError, setFormError] = useState('');
  const [dragging, setDragging] = useState(false);
  const jobInput = useRef(null);
  const resumeInput = useRef(null);

  function addResumes(files) {
    const selected = Array.from(files);
    const valid = selected.filter((file) => ALLOWED_RESUMES.has(extension(file)));
    setFormError(valid.length !== selected.length ? 'Only PDF and DOCX resumes can be added.' : '');
    setResumes((current) => [...current, ...valid.filter((file) => !current.some((saved) => saved.name === file.name && saved.size === file.size && saved.lastModified === file.lastModified))]);
  }

  async function uploadJob(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (extension(file) !== 'txt') {
      setFormError('Use a .txt file for the job description.');
      return;
    }
    try {
      setJobDescription(await file.text());
      setJobFile(file.name);
      setFormError('');
    } catch {
      setFormError('The job description file could not be read.');
    }
  }

  function submit(event) {
    event.preventDefault();
    if (!jobDescription.trim()) return setFormError('Enter a job description or upload a .txt file.');
    if (!resumes.length) return setFormError('Add at least one PDF or DOCX resume.');
    setFormError('');
    onAnalyze(jobDescription, resumes);
  }

  return <form className="analysis-form" onSubmit={submit}>
    <section className="form-card">
      <div className="section-title"><div><p className="eyebrow">Role requirements</p><h2>Job description</h2></div><button className="text-button" type="button" onClick={() => jobInput.current?.click()}>Upload .txt</button></div>
      <input ref={jobInput} className="visually-hidden" type="file" accept=".txt,text/plain" onChange={uploadJob} />
      <textarea value={jobDescription} onChange={(event) => { setJobDescription(event.target.value); setJobFile(''); }} placeholder="Paste the job description here..." aria-label="Job description" />
      {jobFile && <p className="file-note">Loaded from {jobFile}</p>}
    </section>
    <section className="form-card">
      <div className="section-title"><div><p className="eyebrow">Candidate documents</p><h2>Resumes</h2></div><span className="count-label">{resumes.length} selected</span></div>
      <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragEnter={(event) => { event.preventDefault(); setDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); addResumes(event.dataTransfer.files); }}>
        <div className="upload-symbol" aria-hidden="true">+</div><strong>Drag and drop resumes here</strong><span>PDF or DOCX, up to 10 MB each</span>
        <button className="secondary-button" type="button" onClick={() => resumeInput.current?.click()}>Choose files</button>
        <input ref={resumeInput} className="visually-hidden" type="file" accept=".pdf,.docx" multiple onChange={(event) => { addResumes(event.target.files); event.target.value = ''; }} />
      </div>
      {resumes.length > 0 && <ul className="file-list">{resumes.map((resume) => <li key={`${resume.name}-${resume.lastModified}`}><span className="type-label">{extension(resume).toUpperCase()}</span><span className="file-name">{resume.name}</span><span className="file-size">{fileSize(resume.size)}</span><button type="button" className="remove-button" aria-label={`Remove ${resume.name}`} title="Remove resume" onClick={() => setResumes((current) => current.filter((file) => file !== resume))}>x</button></li>)}</ul>}
    </section>
    {(formError || apiError) && <p className="error-message" role="alert">{formError || apiError}</p>}
    <button className="primary-button submit-button" disabled={loading} type="submit">Analyze resumes</button>
  </form>;
}

function Results({ analysis, onDetail, onStartOver }) {
  const { candidates, failed_resumes: failed, job } = analysis;
  return <section>
    <div className="page-heading"><div><p className="eyebrow">Analysis complete</p><h1>{candidates.length} Candidate{candidates.length === 1 ? '' : 's'} Analyzed</h1><p>Ranked using the existing deterministic 100-point scoring model.</p></div><button className="secondary-button" onClick={onStartOver}>New analysis</button></div>
    {failed.length > 0 && <div className="notice"><strong>{failed.length} file{failed.length === 1 ? '' : 's'} could not be analyzed.</strong><List items={failed.map((file) => `${file.source_file}: ${file.error}`)} empty="" /></div>}
    {!candidates.length ? <div className="empty-state"><h2>No candidate results were returned</h2><p>Review the file errors and try another analysis.</p><button className="primary-button" onClick={onStartOver}>Return to uploads</button></div> : <div className="candidate-grid">{candidates.map((candidate, index) => <article className="candidate-card" key={`${candidate.source_file}-${candidate.candidate_name}`}>
      <div className="card-top"><span>Rank {index + 1}</span><strong className="score-chip">{candidate.final_score}/100</strong></div><h2>{candidate.candidate_name || 'Unnamed candidate'}</h2><p className="source-file">{candidate.source_file}</p>
      <div className="score-bar"><span style={{ width: `${candidate.final_score}%` }} /></div>
      <div className="skill-counts"><p><span>Required skills</span><strong>{candidate.matching_required_skills.length}/{job.required_skills.length}</strong></p><p><span>Preferred skills</span><strong>{candidate.matching_preferred_skills.length}/{job.preferred_skills.length}</strong></p></div>
      <ul className="status-list"><Status value={candidate.experience_requirement_met} label="Experience" /><Status value={candidate.education_requirement_satisfied} label="Education" /><Status value={Boolean(candidate.relevant_experience.length)} label="Relevant experience" /></ul>
      <div className="card-section"><h3>Matched skills</h3><Tags items={[...candidate.matching_required_skills, ...candidate.matching_preferred_skills]} empty="No matching skills found." /></div>
      <div className="card-section"><h3>Missing skills</h3><Tags items={candidate.missing_required_skills} empty="No required skills are missing." /></div>
      <button className="text-button card-action" onClick={() => onDetail(candidate)}>View analysis <span aria-hidden="true">&rarr;</span></button>
    </article>)}</div>}
  </section>;
}

function Detail({ candidate, scoreLimits, onBack }) {
  const skills = [...candidate.matching_required_skills, ...candidate.matching_preferred_skills];
  return <section className="detail-view">
    <button className="back-button" onClick={onBack}>&larr; All candidates</button>
    <div className="detail-heading"><div><p className="eyebrow">Candidate analysis</p><h1>{candidate.candidate_name || 'Unnamed candidate'}</h1><p className="source-file">{candidate.source_file}</p></div><div className="score-total"><strong>{candidate.final_score}</strong><span>out of 100</span></div></div>
    <div className="detail-grid">
      <section className="detail-card full"><h2>Score breakdown</h2><div className="breakdown">{Object.entries(BREAKDOWN_LABELS).map(([key, label]) => <p key={key}><span>{label}</span><strong>{candidate.score_breakdown[key]}/{scoreLimits[key]}</strong></p>)}<p className="total"><span>Total</span><strong>{candidate.final_score}/100</strong></p></div></section>
      <section className="detail-card"><h2>Matching skills</h2><Tags items={skills} empty="No matching skills found." /><h2 className="subheading">Missing skills</h2><List items={candidate.missing_required_skills} empty="No required skills are missing." /></section>
      <section className="detail-card"><h2>Strengths</h2><List items={candidate.strengths} empty="No strengths were generated." /><h2 className="subheading">Resume quality issues</h2><List items={candidate.resume_quality_issues} empty="No quality issues were identified." /></section>
      <section className="detail-card"><h2>Job-specific improvement suggestions</h2><List items={candidate.improvement_suggestions} empty="No improvement suggestions were generated." /></section>
      <section className="detail-card"><h2>Evidence and reasoning</h2><p className="verdict">{candidate.verdict || 'No verdict was returned.'}</p><h3>Relevant experience</h3><List items={candidate.relevant_experience} empty="No relevant experience evidence was returned." /><h3>Resume information</h3><div className="information-grid">{Object.entries(candidate.resume_information).map(([key, present]) => <span key={key} className={present ? 'available' : 'unavailable'}>{present ? 'Present' : 'Missing'} {key}</span>)}</div></section>
    </div>
  </section>;
}

export default function App() {
  const [view, setView] = useState('upload');
  const [analysis, setAnalysis] = useState(null);
  const [candidate, setCandidate] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sampleResumes, setSampleResumes] = useState([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  useEffect(() => {
    getSampleResumes().then(setSampleResumes).catch((sampleError) => setError(sampleError.message)).finally(() => setSamplesLoading(false));
  }, []);
  async function runAnalysis(jobDescription, selectedSamples, resumes) {
    setLoading(true); setError('');
    try { const result = await analyzeResumes(jobDescription, selectedSamples, resumes); setAnalysis(result); setView('results'); } catch (err) { setError(err.message); } finally { setLoading(false); }
  }
  function startOver() { setCandidate(null); setAnalysis(null); setError(''); setView('upload'); }
  return <main className="app-shell"><header className="site-header"><button className="brand" onClick={startOver}><span>AR</span><span><strong>AI Resume Rating</strong><small>AI-powered resume screening and job matching</small></span></button>{view !== 'upload' && <button className="text-button" onClick={startOver}>Start over</button>}</header><div className="page-content">{view === 'upload' && <><section className="intro"><p className="eyebrow">Recruiter screening</p><h1>Screen many resumes quickly.</h1><p>Understand the match and see exactly why each candidate ranks where they do.</p></section>{loading ? <section className="loading-state" aria-live="polite"><span className="spinner" /><h2>Analyzing resumes...</h2><p>The AI Agent is reading documents and evaluating each candidate.</p></section> : <ScreeningForm onAnalyze={runAnalysis} apiError={error} initialResumes={sampleResumes} samplesLoading={samplesLoading} />}</>}{view === 'results' && analysis && <ScreeningResults analysis={analysis} onDetail={(selected) => { setCandidate(selected); setView('detail'); }} onStartOver={startOver} />}{view === 'detail' && candidate && analysis && <Detail candidate={candidate} scoreLimits={analysis.score_limits} onBack={() => setView('results')} />}</div></main>;
}
