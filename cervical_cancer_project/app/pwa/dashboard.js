const state = { file: null, installPrompt: null, status: null, previewUrl: null };
const previewMode = location.pathname.startsWith('/preview/');
const byId = id => document.getElementById(id);
document.body.classList.toggle('preview-mode', previewMode);

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 401) {
    window.location.replace('/login');
    throw new Error('Your session has expired.');
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function setMessage(id, text, error = false, success = false) {
  const element = byId(id);
  element.textContent = text;
  element.className = `message${error ? ' error' : ''}${success ? ' success' : ''}`;
}

function populateModelSelect(models) {
  const select = byId('model-select');
  select.replaceChildren();
  models.forEach(model => {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model === 'resnet50' ? 'ResNet50' : 'EfficientNet-B0';
    select.appendChild(option);
  });
  if (!models.length) {
    const option = document.createElement('option');
    option.textContent = 'No trained checkpoint available';
    select.appendChild(option);
  }
}

async function loadStatus() {
  try {
    let account;
    let status;
    if (previewMode) {
      account = { user: { name: 'Demo Researcher' } };
      status = { available_image_models: ['resnet50', 'efficientnet_b0'], clinical_model_available: true, multimodal_fusion_validated: false };
      byId('preview-badge').hidden = false;
    } else {
      account = await jsonRequest('/api/auth/me');
      status = await jsonRequest('/api/status');
    }
    byId('user-name').textContent = account.user.name;
    state.status = status;
    populateModelSelect(status.available_image_models);
    byId('fusion-status');
    byId('clinical-status').textContent = status.clinical_model_available ? 'Model ready' : 'Training required';
    byId('clinical-button').disabled = !status.clinical_model_available;
    updateAnalyseButton();
  } catch (error) {
    setMessage('image-message', error.message, true);
  }
}

function updateAnalyseButton() {
  const available = previewMode || Boolean(state.status?.available_image_models?.length);
  byId('analyse-button').disabled = !state.file || !available;
}

function makeDemoCard(patient) {
  const riskClass = patient.risk_category.toLowerCase();
  const factorList = patient.clinical_factors.map(factor => `<li>${factor}</li>`).join('');
  const clinicalLabels = patient.clinical_risk_factors.map(factor => `<span class="demo-chip">${factor}</span>`).join('');
  const diseaseText = patient.predicted_disease || 'Not available — cell-class dataset';
  const diseaseProbability = patient.disease_probability == null ? 'Not available' : formatPercent(patient.disease_probability);
  const demoCard = document.createElement('article');
  demoCard.className = `demo-card ${riskClass}`;
  demoCard.innerHTML = `
    <div class="demo-card-head"><div class="demo-name"><h3>${patient.patient_name}</h3><small>Synthetic patient · age ${patient.age} · ${patient.gender}</small></div><div><span class="demo-badge">Demo only</span><span class="risk-badge ${riskClass}">${patient.risk_category} risk</span></div></div>
    <div class="demo-card-body">
      <p class="demo-section-label">Patient information</p>
      <div class="demo-profile"><div class="demo-field full"><span>Clinical risk factors</span><div class="demo-chips">${clinicalLabels}</div></div><div class="demo-field"><span>Symptoms</span><strong>${patient.symptoms}</strong></div><div class="demo-field"><span>Screening history</span><strong>${patient.screening_history}</strong></div></div>
      <div class="demo-divider"></div>
      <p class="demo-section-label">Image classification result</p>
      <div class="demo-image-result"><div><div class="cell-class-box"><span>Predicted cell class</span><strong>${patient.predicted_cell_class}</strong></div><div class="model-label">Selected model: <strong>${patient.selected_model}</strong></div><div class="model-label">Selected model confidence: <strong>${formatPercent(patient.model_confidence)}</strong></div><div class="model-label">Predicted disease: <strong>${diseaseText}</strong></div><div class="model-label">Disease probability: <strong>${diseaseProbability}</strong></div><div class="model-label">Disease classification result: <strong>${patient.disease_classification_result}</strong></div></div><div class="confidence-grid"><div class="confidence-line"><span>ResNet50 confidence</span><strong>${formatPercent(patient.resnet50_confidence)}</strong></div><div class="confidence-line"><span>EfficientNet-B0 confidence</span><strong>${formatPercent(patient.efficientnet_b0_confidence)}</strong></div><div class="confidence-line"><span>Ensemble confidence</span><strong>${formatPercent(patient.ensemble_confidence)}</strong></div><div class="confidence-line"><span>Final prediction</span><strong>${patient.predicted_cell_class}</strong></div></div></div>
      <div class="demo-clinical"><div class="risk-score"><span>Early-risk probability</span><strong>${formatPercent(patient.early_risk_probability)}</strong><small>${patient.risk_category} category</small></div><div class="demo-factors"><strong>Main contributing factors</strong><ul>${factorList}</ul></div></div>
      <div class="gradcam-demo"><img src="${patient.gradcam_path}" alt="Synthetic Grad-CAM explanation for ${patient.patient_name}"><div><strong>Grad-CAM explanation</strong><small>Illustrative attention overlay for this synthetic demonstration; it is not evidence from a real patient image.</small></div></div>
      <div class="demo-interpretation"><strong>Final research interpretation</strong><p>${patient.result_summary}</p></div>
      <span class="demo-disclaimer">${patient.disclaimer}</span>
    </div>`;
  return demoCard;
}

async function loadDemoPatients() {
  const grid = byId('demo-grid');
  try {
    const result = await jsonRequest('/api/demo/patients');
    grid.replaceChildren(...result.patients.map(makeDemoCard));
    byId('patient-count').textContent = result.patients.length;
    byId('prediction-count').textContent = result.patients.length;
    byId('high-risk-count').textContent = result.patients.filter(patient => patient.risk_category === 'High').length;
    byId('low-risk-count').textContent = result.patients.filter(patient => patient.risk_category === 'Low').length;
  } catch (error) {
    grid.innerHTML = '<div class="demo-loading surface">Synthetic demo records could not be loaded.</div>';
  }
}

byId('image-input').addEventListener('change', event => {
  state.file = event.target.files[0] || null;
  if (!state.file) return updateAnalyseButton();
  if (!state.file.type.startsWith('image/') || state.file.size > 10 * 1024 * 1024) {
    state.file = null;
    event.target.value = '';
    setMessage('image-message', 'Choose a valid image smaller than 10 MB.', true);
    updateAnalyseButton();
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(state.file);
  byId('preview').src = state.previewUrl;
  byId('preview').style.display = 'block';
  byId('preview-empty').hidden = true;
  setMessage('image-message', 'Image ready for analysis.');
  updateAnalyseButton();
});

byId('analyse-button').addEventListener('click', async () => {
  if (previewMode) {
    setMessage('image-message', 'Visual preview only. Sign in to run a real model inference.', false, true);
    return;
  }
  const form = new FormData();
  form.append('model', byId('model-select').value);
  form.append('image', state.file);
  setMessage('image-message', 'Running model and Grad-CAM…');
  byId('analyse-button').disabled = true;
  try {
    const result = await jsonRequest('/api/image', { method: 'POST', body: form });
    byId('prediction').textContent = result.prediction.replace(/^im_/, '');
    byId('confidence').textContent = `${result.confidence.toFixed(2)}% confidence · ${result.model}`;
    byId('gradcam').src = result.gradcam_overlay;
    const list = byId('probability-list');
    list.replaceChildren(...result.probabilities.map(item => {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `<div class="prob-label"><span>${item.class_name.replace(/^im_/, '')}</span><strong>${item.probability.toFixed(1)}%</strong></div><div class="bar"><i style="width:${item.probability}%"></i></div>`;
      return row;
    }));
    byId('result-empty').hidden = true;
    byId('image-results').hidden = false;
    setMessage('image-message', 'Analysis complete.', false, true);
  } catch (error) {
    setMessage('image-message', error.message, true);
  } finally {
    updateAnalyseButton();
  }
});

byId('clinical-button').addEventListener('click', async () => {
  if (previewMode) {
    setMessage('clinical-result', 'Visual preview only. Sign in to run the independent clinical model.', false, true);
    return;
  }
  const payload = { Age: Number(byId('age').value), Smokes: Number(byId('smokes').value), 'STDs:HPV': Number(byId('hpv').value), 'Hormonal Contraceptives': Number(byId('contraceptives').value) };
  try {
    const result = await jsonRequest('/api/clinical', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    setMessage('clinical-result', `Independent early-risk estimate: ${result.early_risk_probability.toFixed(1)}%. ${result.provided_feature_count} inputs provided; ${result.imputed_feature_count} imputed. ${result.note}`, false, true);
  } catch (error) {
    setMessage('clinical-result', error.message, true);
  }
});

window.addEventListener('beforeinstallprompt', event => {
  event.preventDefault();
  state.installPrompt = event;
  byId('install-button').hidden = false;
});

byId('install-button').addEventListener('click', async () => {
  if (!state.installPrompt) return;
  state.installPrompt.prompt();
  await state.installPrompt.userChoice;
  state.installPrompt = null;
  byId('install-button').hidden = true;
});

byId('logout-button').addEventListener('click', async () => {
  if (previewMode) return window.location.assign('/preview/login');
  await jsonRequest('/api/auth/logout', { method: 'POST' });
  window.location.assign('/login');
});

function closeSidebar() {
  byId('sidebar').classList.remove('open');
  byId('sidebar-overlay').classList.remove('open');
}

byId('sidebar-toggle').addEventListener('click', () => {
  byId('sidebar').classList.toggle('open');
  byId('sidebar-overlay').classList.toggle('open');
});
byId('sidebar-overlay').addEventListener('click', closeSidebar);
document.querySelectorAll('[data-section-link]').forEach(link => link.addEventListener('click', closeSidebar));

if ('serviceWorker' in navigator && !previewMode) navigator.serviceWorker.register('/service-worker.js').catch(() => {});
loadStatus();
loadDemoPatients().then(() => {
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView({ block: 'start' });
});
