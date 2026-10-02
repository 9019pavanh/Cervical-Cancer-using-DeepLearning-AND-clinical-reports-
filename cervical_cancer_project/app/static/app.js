const appState = { viewMode: 'overlay', latestImageResult: null, latestClinicalResult: null };

document.addEventListener('DOMContentLoaded', () => {
  initDropzone();
  toggleSmokingInputs();
});

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(button => button.classList.remove('active'));
  document.querySelectorAll('.tab-section').forEach(section => section.classList.remove('active'));
  document.getElementById(`tab-btn-${tabId}`)?.classList.add('active');
  document.getElementById(`tab-${tabId}`)?.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function initDropzone() {
  const zone = document.getElementById('dropzone');
  const input = document.getElementById('image-file-input');
  if (!zone || !input) return;
  ['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => {
    event.preventDefault();
    zone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => {
    event.preventDefault();
    zone.classList.remove('dragover');
  }));
  zone.addEventListener('drop', event => event.dataTransfer.files[0] && analyzeFile(event.dataTransfer.files[0]));
  input.addEventListener('change', () => input.files[0] && analyzeFile(input.files[0]));
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function analyzeFile(file) {
  if (!file.type.startsWith('image/')) return alert('Please select a valid image.');
  const form = new FormData();
  form.append('file', file);
  await runImageRequest({ method: 'POST', body: form });
}

async function loadSample(sampleId) {
  await runImageRequest({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_id: sampleId })
  });
}

async function runImageRequest(options) {
  showImageLoading(true);
  try {
    const result = await requestJson('/api/predict-image', options);
    renderImageResult(result);
  } catch (error) {
    alert(`Image analysis failed: ${error.message}`);
  } finally {
    showImageLoading(false);
  }
}

function showImageLoading(show) {
  const loader = document.getElementById('image-loading-spinner');
  if (loader) loader.hidden = !show;
}

function setViewMode(mode) {
  appState.viewMode = mode;
  document.querySelectorAll('.view-btn').forEach(button => button.classList.remove('active'));
  document.getElementById(`btn-view-${mode}`)?.classList.add('active');
  updateImageViewer();
}

function updateImageViewer() {
  const image = document.getElementById('main-cytology-img');
  const views = appState.latestImageResult?.visualizations;
  if (image && views) image.src = views[appState.viewMode];
}

function renderImageResult(result) {
  appState.latestImageResult = result;
  updateImageViewer();
  const prediction = result.prediction;
  document.getElementById('predicted-cell-title').textContent = prediction.class_label;
  document.getElementById('predicted-confidence-sub').textContent = `Confidence ${prediction.confidence.toFixed(2)}% · ResNet-18 with 3-view TTA`;
  const badge = document.getElementById('bethesda-stage-badge');
  badge.textContent = prediction.short_label;
  badge.style.color = result.staging?.severity_color || 'var(--cyan)';

  const container = document.getElementById('class-probabilities-list');
  container.replaceChildren(...result.probabilities.map(item => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    const label = document.createElement('span');
    label.textContent = item.label;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('span');
    fill.style.width = `${item.probability}%`;
    bar.appendChild(fill);
    const value = document.createElement('strong');
    value.textContent = `${item.probability.toFixed(1)}%`;
    row.append(label, bar, value);
    return row;
  }));

  const comparison = document.getElementById('image-model-comparison');
  comparison.replaceChildren(...result.model_comparison.map(model => {
    const card = document.createElement('div');
    card.className = 'output-card';
    const name = document.createElement('strong');
    name.textContent = model.model_name;
    const output = document.createElement('span');
    output.textContent = `${model.predicted_class} · ${model.confidence.toFixed(1)}%`;
    card.append(name, output);
    return card;
  }));
}

function toggleSmokingInputs() {
  const enabled = Number(document.getElementById('inp-smokes')?.value) === 1;
  ['inp-smokes-years', 'inp-smokes-packs'].forEach(id => {
    const field = document.getElementById(id);
    if (!field) return;
    field.disabled = !enabled;
    if (!enabled) field.value = 0;
  });
}

function numberValue(id) {
  return Number(document.getElementById(id).value);
}

function clinicalPayload() {
  return {
    Age: numberValue('inp-age'),
    'First sexual intercourse': numberValue('inp-first-intercourse'),
    'Number of sexual partners': numberValue('inp-partners'),
    'Num of pregnancies': numberValue('inp-pregnancies'),
    Smokes: numberValue('inp-smokes'),
    'Smokes (years)': numberValue('inp-smokes-years'),
    'Smokes (packs/year)': numberValue('inp-smokes-packs'),
    'Hormonal Contraceptives': numberValue('inp-hc'),
    'Hormonal Contraceptives (years)': numberValue('inp-hc-years'),
    IUD: numberValue('inp-iud'),
    'IUD (years)': numberValue('inp-iud-years'),
    STDs: numberValue('inp-stds'),
    'STDs:HPV': numberValue('inp-std-hpv'),
    'STDs:HIV': numberValue('inp-std-hiv')
  };
}

async function submitClinicalForm() {
  const button = document.getElementById('btn-calculate-risk');
  if (button) button.disabled = true;
  try {
    const result = await requestJson('/api/predict-clinical', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(clinicalPayload())
    });
    appState.latestClinicalResult = result;
    renderClinicalResult(result);
    return result;
  } catch (error) {
    alert(`Clinical risk calculation failed: ${error.message}`);
    throw error;
  } finally {
    if (button) button.disabled = false;
  }
}

function renderClinicalResult(result) {
  document.getElementById('risk-score-val').textContent = `${result.risk_percentage.toFixed(1)}%`;
  document.getElementById('clinical-risk-badge').textContent = result.risk_level;
  document.getElementById('clinical-summary-text').textContent = result.summary_statement;
  const drivers = document.getElementById('risk-drivers-list');
  const driverItems = result.key_risk_drivers.length ? result.key_risk_drivers : [{ label: 'No dominant factors', clinical_note: 'No high-impact supplied factor was identified.' }];
  drivers.replaceChildren(...driverItems.map(item => textCard('driver', item.label, item.clinical_note)));
  const suggestions = document.getElementById('suggestions-list');
  suggestions.replaceChildren(...result.clinical_suggestions.map(item => textCard('suggestion', item.title, item.detail)));
}

function textCard(className, title, detail) {
  const card = document.createElement('div');
  card.className = className;
  const heading = document.createElement('strong');
  heading.textContent = title;
  const body = document.createElement('span');
  body.textContent = detail;
  card.append(heading, body);
  return card;
}

async function generateAndShowReport() {
  if (!appState.latestImageResult) {
    alert('Complete the cytology module first.');
    switchTab('cytology');
    return;
  }
  try {
    const clinical = appState.latestClinicalResult || await submitClinicalForm();
    const report = await requestJson('/api/generate-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_result: appState.latestImageResult, clinical_result: clinical })
    });
    renderReport(report);
    switchTab('report');
  } catch (error) {
    alert(`Hybrid result failed: ${error.message}`);
  }
}

function renderReport(report) {
  const fusion = report.hybrid_fusion;
  document.getElementById('rep-triage-title').textContent = report.overall_tier;
  document.getElementById('rep-urgency-val').textContent = report.urgency;
  document.getElementById('rep-hybrid-score').textContent = `${fusion.score.toFixed(1)}%`;
  document.getElementById('rep-cytology-thumb').src = appState.latestImageResult.visualizations.overlay;
  document.getElementById('rep-cell-class').textContent = report.cytology_summary.cell_class;
  document.getElementById('rep-ai-conf').textContent = `${report.cytology_summary.confidence.toFixed(1)}%`;
  document.getElementById('rep-cytology-desc').textContent = `Image abnormality probability: ${fusion.image_abnormality_probability.toFixed(1)}%`;
  document.getElementById('rep-pat-risk').textContent = `${fusion.clinical_risk_probability.toFixed(1)}%`;
  document.getElementById('rep-clinical-summary').textContent = report.clinical_summary;
  const actions = document.getElementById('rep-actions-container');
  actions.replaceChildren(...report.action_items.map(item => textCard('action', item.title, item.detail)));
}
