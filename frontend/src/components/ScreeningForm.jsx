import { useRef, useState } from 'react';
import { getSampleResumeUrl } from '../api';

const ALLOWED_RESUMES = new Set(['pdf', 'docx']);

function extension(name) {
  return name.split('.').pop()?.toUpperCase() || 'FILE';
}

function size(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function entryName(entry) {
  return entry.kind === 'sample' ? entry.name : entry.file.name;
}

function entrySize(entry) {
  return entry.kind === 'sample' ? entry.size : entry.file.size;
}

function sampleTitle(filename) {
  return filename.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export default function ScreeningForm({ initialResumes, samplesLoading, apiError, onAnalyze }) {
  const [jobDescription, setJobDescription] = useState('');
  const [jobFile, setJobFile] = useState('');
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const jobInput = useRef(null);
  const resumeInput = useRef(null);

  function sampleIsSelected(sample) {
    return entries.some((entry) => entry.kind === 'sample' && entry.name === sample.name);
  }

  function addSample(sample) {
    setEntries((current) => current.some((entry) => entry.kind === 'sample' && entry.name === sample.name) ? current : [...current, { ...sample, kind: 'sample' }]);
  }

  function addFiles(files) {
    const selected = Array.from(files);
    const valid = selected.filter((file) => ALLOWED_RESUMES.has(file.name.split('.').pop()?.toLowerCase()));
    setError(valid.length === selected.length ? '' : 'Only PDF and DOCX resumes can be added.');
    setEntries((current) => [
      ...current,
      ...valid
        .filter((file) => !current.some((entry) => entryName(entry) === file.name && entrySize(entry) === file.size))
        .map((file) => ({ kind: 'upload', file })),
    ]);
  }

  async function loadJobFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.name.split('.').pop()?.toLowerCase() !== 'txt') {
      setError('Use a .txt file for the job description.');
      return;
    }
    try {
      setJobDescription(await file.text());
      setJobFile(file.name);
      setError('');
    } catch {
      setError('The job description file could not be read.');
    }
  }

  function submit(event) {
    event.preventDefault();
    if (!jobDescription.trim()) return setError('Enter a job description or upload a .txt file.');
    if (!entries.length) return setError('Add at least one candidate before starting analysis.');
    setError('');
    onAnalyze(
      jobDescription,
      entries.filter((entry) => entry.kind === 'sample'),
      entries.filter((entry) => entry.kind === 'upload').map((entry) => entry.file),
    );
  }

  return <form className="analysis-form" onSubmit={submit}>
    <section className="form-card">
      <div className="section-title"><div><p className="eyebrow">Role requirements</p><h2>Job description</h2></div><button className="text-button" type="button" onClick={() => jobInput.current?.click()}>Upload .txt</button></div>
      <input ref={jobInput} className="visually-hidden" type="file" accept=".txt,text/plain" onChange={loadJobFile} />
      <textarea value={jobDescription} onChange={(event) => { setJobDescription(event.target.value); setJobFile(''); }} placeholder="Paste the job description here..." aria-label="Job description" />
      {jobFile && <p className="file-note">Loaded from {jobFile}</p>}
    </section>

    <section className="form-card">
      <div className="section-title"><div><p className="eyebrow">Your resumes</p><h2>Upload candidates</h2></div><span className="count-label">Added files join selected candidates</span></div>
      <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragEnter={(event) => { event.preventDefault(); setDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files); }}>
        <div className="upload-symbol" aria-hidden="true">+</div><strong>Drag and drop resumes here</strong><span>PDF or DOCX, up to 10 MB each</span>
        <button className="secondary-button" type="button" onClick={() => resumeInput.current?.click()}>Choose files</button>
        <input ref={resumeInput} className="visually-hidden" type="file" accept=".pdf,.docx" multiple onChange={(event) => { addFiles(event.target.files); event.target.value = ''; }} />
      </div>
    </section>

    <section className="form-card selected-card">
      <div className="section-title"><div><p className="eyebrow">Analysis pool</p><h2>Selected candidates</h2></div><span className="count-label">{entries.length} selected</span></div>
      <p className="batch-note">Only these resumes are sent to the analysis pipeline.</p>
      {entries.length ? <ul className="file-list">{entries.map((entry) => <li key={`${entry.kind}-${entryName(entry)}`}><span className="type-label">{entry.kind === 'sample' ? 'Sample' : extension(entryName(entry))}</span><span className="file-name">{entryName(entry)}</span><span className="file-size">{size(entrySize(entry))}</span><button type="button" className="remove-button" aria-label={`Remove ${entryName(entry)}`} title="Remove resume" onClick={() => setEntries((current) => current.filter((item) => item !== entry))}>x</button></li>)}</ul> : <div className="selected-empty"><strong>No candidates selected</strong><span>Add a sample for a quick test, or upload your own resumes.</span></div>}
    </section>

    {(error || apiError) && <p className="error-message" role="alert">{error || apiError}</p>}
    <button className="primary-button submit-button" type="submit">Analyze candidates</button>

    <section className="form-card demo-card compact-demo-card">
      <div className="section-title"><div><p className="eyebrow">Optional demo</p><h2>Sample resumes</h2></div><span className="count-label">Try the demo</span></div>
      {samplesLoading ? <p className="muted sample-loading">Loading repository sample resumes...</p> : <div className="sample-grid compact-sample-grid">{initialResumes.map((sample) => <article className="sample-item" key={sample.name}><div><h3>{sampleTitle(sample.name)}</h3><p>{extension(sample.name)} · {size(sample.size)}</p></div><div className="sample-actions"><button className="secondary-button" type="button" onClick={() => window.open(getSampleResumeUrl(sample.name), '_blank', 'noopener,noreferrer')}>View</button><button className="sample-add" type="button" disabled={sampleIsSelected(sample)} onClick={() => addSample(sample)}>{sampleIsSelected(sample) ? 'Added' : 'Add'}</button></div></article>)}</div>}
    </section>
  </form>;
}
