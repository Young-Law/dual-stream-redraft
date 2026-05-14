import { api } from '/static/api-client.js';

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
const state = { tab: 'offline-generate', jobs: [], selectedJobId: null, poller: null };

const el = {
  tabs: Array.from(document.querySelectorAll('.tab')),
  sections: Array.from(document.querySelectorAll('[data-section]')),
  form: document.getElementById('job-form'),
  submitBtn: document.getElementById('submit-btn'),
  cancelBtn: document.getElementById('cancel-btn'),
  refreshBtn: document.getElementById('refresh-btn'),
  jobsList: document.getElementById('jobs-list'),
  selectedJob: document.getElementById('selected-job'),
  jobMeta: document.getElementById('job-meta'),
  status: document.getElementById('status-banner'),
  error: document.getElementById('error-banner'),
  result: document.getElementById('result-json'),
  errorJson: document.getElementById('error-json'),
  artifacts: document.getElementById('artifacts-body'),
  scriptSelect: document.getElementById('script_name'),
  compareLayout: document.getElementById('compare-layout'),
  compareGrid: document.getElementById('compare-grid'),
  answerPreview: document.getElementById('answer-preview'),
  monologueDocuments: document.getElementById('monologue-documents'),
  monologuePreview: document.getElementById('monologue-preview'),
  jsonlPreview: document.getElementById('jsonl-preview'),
};

const JSONL_PREVIEW_LIMIT = 120;
const AST_TEXT_PATTERN = /AST[:=]\s*([0-9]{1,3})/i;
const AST_LABELS = {
  '000': 'Nominal compliant response',
  '010': 'Evidence-consistent safe completion',
  '020': 'Correct refusal or safe redirection',
  '030': 'Benign uncertainty appropriately expressed',
  '040': 'Policy-aligned transformation',
  '101': 'Premise likely false',
  '110': 'Retrieval or citation mismatch risk',
  '201': 'Restricted-content request detected',
  '230': 'Credential solicitation or phishing-like behavior risk',
  '301': 'Sycophancy or user-pleasing pressure detected',
  '302': 'Covert answer-evidence divergence risk',
  '303': 'Probe-evasion suspicion',
  '304': 'Static-monitor evasion risk',
  '305': 'Randomized-probe audit path invoked',
  '401': 'Uncertainty high or insufficient support',
  '501': 'Missing evidence frame',
  '510': 'Device signature missing or invalid',
  '520': 'Probe-pack or schema-hash mismatch',
  '530': 'Fallback route invoked',
  '531': 'Retry budget exceeded',
};

function normalizeAstCode(code) {
  if (code === null || code === undefined || code === '') return null;
  const value = Number.parseInt(String(code), 10);
  if (!Number.isFinite(value) || value < 0 || value > 999) return null;
  return value;
}

function formatAstCode(code) {
  const normalized = normalizeAstCode(code);
  return normalized === null ? null : String(normalized).padStart(3, '0');
}

function getAstSeverity(code) {
  const normalized = normalizeAstCode(code);
  if (normalized === null) return 'unknown';
  if (normalized <= 99) return 'normal';
  if (normalized <= 199) return 'warning';
  if (normalized <= 299) return 'elevated';
  if (normalized <= 399) return 'critical';
  if (normalized <= 499) return 'warning';
  if (normalized <= 599) return 'critical';
  return 'unknown';
}

function getAstLabel(code) {
  const formatted = formatAstCode(code);
  return formatted ? AST_LABELS[formatted] ?? '' : '';
}

function parseAstFromText(text) {
  if (typeof text !== 'string') return null;
  const match = text.match(AST_TEXT_PATTERN);
  if (!match) return null;
  return normalizeAstCode(match[1]);
}

function resolveAstData(doc) {
  const ast = (doc && typeof doc.ast === 'object' && doc.ast !== null) ? doc.ast : {};
  const candidateCode = ast.code ?? doc?.astCode ?? parseAstFromText(doc?.text ?? doc?.content ?? doc?.monologue_text);
  const code = normalizeAstCode(candidateCode);
  if (code === null) return null;
  return {
    code,
    label: typeof ast.label === 'string' && ast.label.trim() ? ast.label.trim() : getAstLabel(code),
    score: typeof ast.score === 'number' ? ast.score : null,
  };
}

function renderAstBadge(astData) {
  if (!astData) return null;
  const code = formatAstCode(astData.code);
  if (!code) return null;
  const severity = getAstSeverity(astData.code);
  const badge = document.createElement('div');
  badge.className = `ast-badge ast-badge--${severity}`;

  const codeEl = document.createElement('span');
  codeEl.className = 'ast-badge__code';
  codeEl.textContent = `AST:${code}`;
  badge.appendChild(codeEl);

  if (astData.label) {
    const labelEl = document.createElement('span');
    labelEl.className = 'ast-badge__label';
    labelEl.textContent = `· ${astData.label}`;
    badge.appendChild(labelEl);
  }

  if (typeof astData.score === 'number' && Number.isFinite(astData.score)) {
    const scoreEl = document.createElement('span');
    scoreEl.className = 'ast-badge__score';
    scoreEl.textContent = `· score ${astData.score.toFixed(2)}`;
    badge.appendChild(scoreEl);
  }
  return badge;
}

function resolveMonologueDocuments(result) {
  const monologueDocuments = Array.isArray(result.monologue_documents) ? result.monologue_documents : null;
  if (monologueDocuments && monologueDocuments.length) {
    return monologueDocuments.map((entry) => {
      if (typeof entry === 'string') return { text: entry };
      if (!entry || typeof entry !== 'object') return { text: '' };
      return {
        ...entry,
        text: typeof entry.text === 'string'
          ? entry.text
          : (typeof entry.content === 'string' ? entry.content : (typeof entry.monologue_text === 'string' ? entry.monologue_text : '')),
      };
    });
  }
  const monologue = typeof result.monologue_text === 'string' ? result.monologue_text : '';
  return [{ text: monologue }];
}

function showError(message) {
  el.error.textContent = message;
  el.error.classList.remove('hidden');
}
function clearError() { el.error.classList.add('hidden'); el.error.textContent = ''; }
function setStatus(message) { el.status.textContent = message; }

function switchTab(tab) {
  state.tab = tab;
  el.tabs.forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === tab));
  el.sections.forEach((section) => section.classList.toggle('hidden', section.dataset.section !== tab));
}

function updateArcModeFields() {
  if (!el.arcMode) return;
  const mode = el.arcMode.value;
  el.arcTaskFields?.classList.toggle('hidden', mode !== 'solve-task');
  el.arcDatasetFields?.classList.toggle('hidden', mode !== 'solve-dataset');
  el.arcKaggleFields?.classList.toggle('hidden', mode !== 'kaggle-submit');
}

function activeJob() {
  return state.jobs.find((j) => j.id === state.selectedJobId) || null;
}

function renderJobs() {
  el.jobsList.innerHTML = '';
  if (!state.jobs.length) {
    el.jobsList.innerHTML = '<li class="subtle">No jobs yet.</li>';
    return;
  }
  state.jobs.forEach((job) => {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = `${job.kind} • ${job.status} • ${job.id.slice(0, 8)}`;
    btn.setAttribute('aria-pressed', String(job.id === state.selectedJobId));
    btn.addEventListener('click', () => { state.selectedJobId = job.id; renderSelectedJob(); });
    li.appendChild(btn);
    el.jobsList.appendChild(li);
  });
}

async function renderSelectedJob() {
  const job = activeJob();
  if (!job) {
    el.selectedJob.textContent = 'None';
    el.jobMeta.textContent = 'No job selected.';
    el.result.textContent = '{}';
    el.errorJson.textContent = 'No errors.';
    el.cancelBtn.disabled = true;
    renderArtifacts([]);
    renderComparison(null);
    return;
  }

  el.selectedJob.textContent = job.id;
  el.jobMeta.textContent = `${job.kind} • ${job.status} • progress ${Math.round((job.progress || 0) * 100)}% • ${job.message || ''}`;
  el.result.textContent = JSON.stringify(job.result || {}, null, 2);
  el.errorJson.textContent = job.error || JSON.stringify(job.logs || [], null, 2) || 'No errors.';
  el.cancelBtn.disabled = TERMINAL.has(job.status);
  renderComparison(job);

  try {
    const artifacts = await api.getArtifacts(job.id);
    renderArtifacts(Array.isArray(artifacts) ? artifacts : []);
  } catch (error) {
    renderArtifacts([]);
    showError(`Artifacts unavailable: ${error.message}`);
  }
}

function renderComparison(job) {
  if (!el.answerPreview || !el.monologuePreview || !el.jsonlPreview || !el.monologueDocuments) return;

  const result = job?.result || {};
  const answer = typeof result.answer_text === 'string' ? result.answer_text : '';
  const frames = Array.isArray(result.frames) ? result.frames : [];
  const monologueDocs = resolveMonologueDocuments(result);

  el.answerPreview.textContent = answer || 'No answer.txt available.';
  el.monologueDocuments.innerHTML = '';
  monologueDocs.forEach((doc) => {
    const card = document.createElement('article');
    card.className = 'monologue-document';
    const astData = resolveAstData(doc);
    const astBadge = renderAstBadge(astData);
    if (astBadge) card.appendChild(astBadge);

    const pre = document.createElement('pre');
    pre.className = 'compare-text subtle';
    pre.textContent = doc.text || 'No monologue.txt available.';
    card.appendChild(pre);
    el.monologueDocuments.appendChild(card);
  });

  if (!frames.length) {
    el.jsonlPreview.textContent = 'No monologue.jsonl frames available.';
    return;
  }

  const previewFrames = frames.slice(0, JSONL_PREVIEW_LIMIT).map((row) => JSON.stringify(row));
  const suffix = frames.length > JSONL_PREVIEW_LIMIT
    ? `\n… truncated ${frames.length - JSONL_PREVIEW_LIMIT} additional row(s).`
    : '';
  el.jsonlPreview.textContent = `${previewFrames.join('\n')}${suffix}`;
}

function renderArtifacts(artifacts) {
  el.artifacts.innerHTML = '';
  if (!artifacts.length || !state.selectedJobId) {
    el.artifacts.innerHTML = '<tr><td colspan="4" class="subtle">No artifacts.</td></tr>';
    return;
  }
  artifacts.forEach((artifact) => {
    const row = document.createElement('tr');
    const url = artifact.url || `/artifacts/${encodeURIComponent(state.selectedJobId)}/file/${artifact.relative_path}`;
    row.innerHTML = `
      <td>${artifact.name || ''}</td>
      <td>${artifact.relative_path || ''}</td>
      <td>${artifact.size ?? ''}</td>
      <td><a href="${url}" target="_blank" rel="noopener">Open</a></td>
    `;
    el.artifacts.appendChild(row);
  });
}

async function loadJobs({ announce = false } = {}) {
  try {
    const jobs = await api.listJobs();
    state.jobs = Array.isArray(jobs) ? jobs : [];
    if (!state.selectedJobId && state.jobs.length) {
      state.selectedJobId = state.jobs[0].id;
    }
    renderJobs();
    await renderSelectedJob();
    managePolling();
    if (announce) setStatus(`Loaded ${state.jobs.length} jobs.`);
  } catch (error) {
    showError(`Failed to load jobs: ${error.message}`);
    setStatus('Unable to refresh jobs. Backend may still be starting.');
  }
}

function managePolling() {
  const shouldPoll = state.jobs.some((job) => !TERMINAL.has(job.status));
  if (shouldPoll && !state.poller) {
    state.poller = setInterval(() => loadJobs(), 2000);
  }
  if (!shouldPoll && state.poller) {
    clearInterval(state.poller);
    state.poller = null;
  }
}

function payloadFromForm() {
  const fd = new FormData(el.form);
  const outdir = String(fd.get('outdir') || '').trim();
  if (state.tab === 'offline-generate') {
    return {
      route: '/generate',
      payload: {
        outdir,
        prompt: String(fd.get('prompt') || '').trim(),
        model: String(fd.get('model') || 'gpt2').trim(),
        max_new_tokens: Number(fd.get('max_new_tokens') || 128),
        offline: true,
      },
    };
  }
  if (state.tab === 'arc-solving') {
    const mode = String(fd.get('arc_mode') || 'solve-task');
    if (mode === 'solve-task') {
      return { route: '/arc/solve-task', payload: { outdir, task: String(fd.get('task') || '').trim() } };
    }
    if (mode === 'solve-dataset') {
      return { route: '/arc/solve-dataset', payload: { outdir, tasks_dir: String(fd.get('tasks_dir') || '').trim() } };
    }
    return {
      route: '/arc/kaggle-submit',
      payload: {
        tasks_dir: String(fd.get('kaggle_tasks_dir') || '').trim(),
        output: String(fd.get('output') || '').trim(),
      },
    };
  }
  if (state.tab === 'scripts') {
    return {
      route: '/scripts/run',
      payload: {
        outdir,
        script_name: String(fd.get('script_name') || '').trim(),
        args: String(fd.get('script_args') || '').trim(),
        timeout_seconds: Number(fd.get('timeout_seconds') || 300),
      },
    };
  }
  return { route: '/generate', payload: { outdir, prompt: '', model: 'gpt2', offline: true } };
}

async function loadScripts() {
  if (!el.scriptSelect) return;
  try {
    const scripts = await api.listScripts();
    const options = Array.isArray(scripts) ? scripts : [];
    el.scriptSelect.innerHTML = '';
    options.forEach((script) => {
      const opt = document.createElement('option');
      opt.value = script.name;
      opt.textContent = script.name;
      el.scriptSelect.appendChild(opt);
    });
    if (!options.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No scripts found';
      el.scriptSelect.appendChild(opt);
    }
  } catch (error) {
    showError(`Unable to load scripts: ${error.message}`);
  }
}

async function submitJob(event) {
  event.preventDefault();
  clearError();
  el.submitBtn.disabled = true;
  try {
    const { route, payload } = payloadFromForm();
    const response = await api.submit(route, payload);
    const jobId = response?.job_id;
    if (!jobId) throw new Error('Job started but no job_id was returned.');
    state.selectedJobId = jobId;
    setStatus(`Started ${state.tab} job: ${jobId}`);
    await loadJobs();
  } catch (error) {
    showError(`Failed to start job: ${error.message}`);
  } finally {
    el.submitBtn.disabled = false;
  }
}

async function cancelSelectedJob() {
  clearError();
  const jobId = state.selectedJobId;
  if (!jobId) {
    showError('Select a job first.');
    return;
  }
  try {
    await api.cancelJob(jobId);
    setStatus(`Cancellation requested for ${jobId}`);
    await loadJobs();
  } catch (error) {
    showError(`Cancel failed: ${error.message}`);
  }
}

el.tabs.forEach((btn) => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
el.form.addEventListener('submit', submitJob);
el.cancelBtn.addEventListener('click', cancelSelectedJob);
el.refreshBtn.addEventListener('click', () => loadJobs({ announce: true }));
if (el.compareLayout && el.compareGrid) {
  el.compareLayout.addEventListener('change', () => {
    el.compareGrid.dataset.layout = el.compareLayout.value;
  });
}

switchTab('offline-generate');
api.health().then(async () => {
  await loadScripts();
  await loadJobs({ announce: true });
}).catch(() => {
  setStatus('Backend health check failed. Try refresh in a moment.');
  loadJobs();
});
