/**
 * app.js
 * ======
 * Frontend Controller for Cervical Cancer AI Diagnostic Suite.
 * Handles image upload, Grad-CAM view modes, Bethesda staging rendering,
 * model comparison showcase, clinical risk prediction, and report generation.
 */

// Global Application State
const appState = {
    currentTab: 'cytology',
    viewMode: 'overlay', // 'overlay' | 'heatmap' | 'original'
    latestImageResult: null,
    latestClinicalResult: null,
    currentSampleId: 'im_Dyskeratotic',
};

// Preset Profiles for Rapid Clinical Testing
const CLINICAL_PRESETS = {
    low: {
        Age: 26,
        'First sexual intercourse': 19,
        'Number of sexual partners': 1,
        'Num of pregnancies': 0,
        Smokes: 0,
        'Smokes (years)': 0,
        'Smokes (packs/year)': 0,
        'Hormonal Contraceptives': 1,
        'Hormonal Contraceptives (years)': 1,
        IUD: 0,
        'IUD (years)': 0,
        STDs: 0,
        'STDs:HPV': 0,
        'STDs:HIV': 0,
    },
    moderate: {
        Age: 35,
        'First sexual intercourse': 17,
        'Number of sexual partners': 3,
        'Num of pregnancies': 2,
        Smokes: 0,
        'Smokes (years)': 0,
        'Smokes (packs/year)': 0,
        'Hormonal Contraceptives': 1,
        'Hormonal Contraceptives (years)': 5,
        IUD: 0,
        'IUD (years)': 0,
        STDs: 1,
        'STDs:HPV': 1,
        'STDs:HIV': 0,
    },
    high: {
        Age: 44,
        'First sexual intercourse': 15,
        'Number of sexual partners': 5,
        'Num of pregnancies': 4,
        Smokes: 1,
        'Smokes (years)': 14,
        'Smokes (packs/year)': 8,
        'Hormonal Contraceptives': 1,
        'Hormonal Contraceptives (years)': 9,
        IUD: 0,
        'IUD (years)': 0,
        STDs: 1,
        'STDs:HPV': 1,
        'STDs:HIV': 0,
    }
};

// Initialize Application on Page Load
document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    // Automatically evaluate the default Dyskeratotic sample so the UI is immediately populated
    loadSample('im_Dyskeratotic');
    // Pre-calculate initial clinical risk with default values
    submitClinicalForm();
});

// ==============================================================================
// TAB NAVIGATION
// ==============================================================================
function switchTab(tabId) {
    appState.currentTab = tabId;

    // Update Tab Buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`tab-btn-${tabId}`);
    if (activeBtn) activeBtn.classList.add('active');

    // Update Sections
    document.querySelectorAll('.tab-section').forEach(sec => sec.classList.remove('active'));
    const activeSection = document.getElementById(`tab-${tabId}`);
    if (activeSection) activeSection.classList.add('active');

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==============================================================================
// IMAGE DROPZONE & UPLOAD HANDLERS
// ==============================================================================
function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('image-file-input');

    if (!dropzone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleImageUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleImageUpload(fileInput.files[0]);
        }
    });
}

function handleImageUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please upload a valid cytology image file (.png, .jpg, .bmp)');
        return;
    }

    showImageLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict-image', {
        method: 'POST',
        body: formData,
    })
    .then(res => res.json())
    .then(data => {
        showImageLoading(false);
        if (data.error) {
            alert('Analysis error: ' + data.error);
            return;
        }
        renderImageAnalysisResult(data);
    })
    .catch(err => {
        showImageLoading(false);
        console.error('Upload error:', err);
        alert('Failed to connect to diagnostic server.');
    });
}

function loadSample(sampleId) {
    appState.currentSampleId = sampleId;
    showImageLoading(true);

    fetch('/api/predict-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_id: sampleId }),
    })
    .then(res => res.json())
    .then(data => {
        showImageLoading(false);
        if (data.error) {
            console.error('Error loading sample:', data.error);
            return;
        }
        renderImageAnalysisResult(data);
    })
    .catch(err => {
        showImageLoading(false);
        console.error('Sample fetch error:', err);
    });
}

function showImageLoading(isLoading) {
    const spinner = document.getElementById('image-loading-spinner');
    if (spinner) {
        spinner.style.display = isLoading ? 'block' : 'none';
    }
}

// ==============================================================================
// GRAD-CAM 3-MODE VIEW SWITCHER
// ==============================================================================
function setViewMode(mode) {
    appState.viewMode = mode;

    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`btn-view-${mode}`);
    if (activeBtn) activeBtn.classList.add('active');

    updateImageViewer();
}

function updateImageViewer() {
    const imgEl = document.getElementById('main-cytology-img');
    const res = appState.latestImageResult;
    if (!imgEl || !res || !res.visualizations) return;

    if (appState.viewMode === 'overlay') {
        imgEl.src = res.visualizations.overlay;
    } else if (appState.viewMode === 'heatmap') {
        imgEl.src = res.visualizations.heatmap;
    } else {
        imgEl.src = res.visualizations.original;
    }
}

// ==============================================================================
// RENDER CYTOLOGY & STAGING RESULTS
// ==============================================================================
function renderImageAnalysisResult(result) {
    appState.latestImageResult = result;
    updateImageViewer();

    const pred = result.prediction;
    const staging = result.staging;

    // 1. Title & Primary Confidence
    const titleEl = document.getElementById('predicted-cell-title');
    const subEl = document.getElementById('predicted-confidence-sub');
    if (titleEl) titleEl.textContent = staging.name || pred.class_label;
    if (subEl) subEl.textContent = `Confidence: ${pred.confidence.toFixed(2)}% • Primary Consensus Diagnosis`;

    // 2. Bethesda Staging Card
    const badgeEl = document.getElementById('bethesda-stage-badge');
    if (badgeEl) {
        badgeEl.textContent = staging.stage_code || staging.bethesda_category;
        badgeEl.className = 'stage-tag';
        if (staging.severity === 'Critical') {
            badgeEl.classList.add('stage-tag-critical');
        } else if (staging.severity === 'Warning') {
            badgeEl.classList.add('stage-tag-warning');
        } else {
            badgeEl.classList.add('stage-tag-normal');
        }
    }

    const valBethesda = document.getElementById('stage-val-bethesda');
    const valCin = document.getElementById('stage-val-cin');
    const valStage = document.getElementById('stage-val-stage');
    const valRisk = document.getElementById('stage-val-risk');
    const valAction = document.getElementById('stage-val-action');

    if (valBethesda) valBethesda.textContent = staging.bethesda_category || 'NILM';
    if (valCin) valCin.textContent = staging.cin_grade || 'Normal';
    if (valStage) valStage.textContent = staging.cancer_stage || 'Stage 0';
    if (valRisk) {
        valRisk.textContent = staging.malignancy_risk || 'Normal Risk';
        valRisk.style.color = staging.severity_color || '#34c759';
    }
    if (valAction) valAction.textContent = staging.immediate_action || 'Routine follow-up';

    // 3. Multi-Model Performance Comparison
    const resnetPredLabel = document.getElementById('resnet-pred-label');
    const effnetPredLabel = document.getElementById('effnet-pred-label');
    const hybridPredLabel = document.getElementById('hybrid-pred-label');

    if (result.model_comparison && result.model_comparison.length >= 3) {
        const m0 = result.model_comparison[0]; // ResNet-18
        const m1 = result.model_comparison[1]; // EfficientNet-B0
        const m2 = result.model_comparison[2]; // Dual Hybrid Model

        if (resnetPredLabel) resnetPredLabel.textContent = `${m0.predicted_class} (${m0.confidence.toFixed(1)}%)`;
        if (effnetPredLabel) effnetPredLabel.textContent = `${m1.predicted_class} (${m1.confidence.toFixed(1)}%)`;
        if (hybridPredLabel) hybridPredLabel.textContent = `${m2.predicted_class} (${m2.confidence.toFixed(1)}%)`;
    }

    // 4. Class Probabilities Progress Bars
    const probContainer = document.getElementById('class-probabilities-list');
    if (probContainer && result.probabilities) {
        probContainer.innerHTML = '';
        result.probabilities.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'prob-item';
            itemDiv.innerHTML = `
                <div class="prob-labels">
                    <span style="color: ${item.is_top ? '#fff' : 'var(--text-secondary)'}; font-weight: ${item.is_top ? '700' : '500'};">
                        ${item.is_top ? '★ ' : ''}${item.label}
                    </span>
                    <span style="font-family: var(--font-mono); color: ${item.is_top ? 'var(--cyan-accent)' : 'var(--text-muted)'};">
                        ${item.probability.toFixed(1)}%
                    </span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: ${item.probability}%; ${item.is_top ? '' : 'background: rgba(255,255,255,0.2);'}"></div>
                </div>
            `;
            probContainer.appendChild(itemDiv);
        });
    }
}

// ==============================================================================
// CLINICAL RISK PREDICTION FORM & ENGINE
// ==============================================================================
function toggleSmokingInputs() {
    const smokesVal = parseInt(document.getElementById('inp-smokes').value, 10);
    const yearsInput = document.getElementById('inp-smokes-years');
    const packsInput = document.getElementById('inp-smokes-packs');

    if (smokesVal === 0) {
        yearsInput.value = '0';
        packsInput.value = '0';
        yearsInput.disabled = true;
        packsInput.disabled = true;
    } else {
        yearsInput.disabled = false;
        packsInput.disabled = false;
        if (yearsInput.value === '0') yearsInput.value = '5';
        if (packsInput.value === '0') packsInput.value = '2';
    }
}

function applyPreset(presetKey) {
    const p = CLINICAL_PRESETS[presetKey];
    if (!p) return;

    document.getElementById('inp-age').value = p.Age;
    document.getElementById('inp-first-intercourse').value = p['First sexual intercourse'];
    document.getElementById('inp-partners').value = p['Number of sexual partners'];
    document.getElementById('inp-pregnancies').value = p['Num of pregnancies'];

    document.getElementById('inp-smokes').value = p.Smokes;
    toggleSmokingInputs();
    document.getElementById('inp-smokes-years').value = p['Smokes (years)'];
    document.getElementById('inp-smokes-packs').value = p['Smokes (packs/year)'];

    document.getElementById('inp-hc').value = p['Hormonal Contraceptives'];
    document.getElementById('inp-hc-years').value = p['Hormonal Contraceptives (years)'];
    document.getElementById('inp-iud').value = p.IUD;
    document.getElementById('inp-iud-years').value = p['IUD (years)'];

    document.getElementById('inp-stds').value = p.STDs;
    document.getElementById('inp-std-hpv').value = p['STDs:HPV'];
    document.getElementById('inp-std-hiv').value = p['STDs:HIV'];

    submitClinicalForm();
}

function getClinicalFormData() {
    return {
        Age: parseFloat(document.getElementById('inp-age').value) || 30,
        'First sexual intercourse': parseFloat(document.getElementById('inp-first-intercourse').value) || 18,
        'Number of sexual partners': parseFloat(document.getElementById('inp-partners').value) || 1,
        'Num of pregnancies': parseFloat(document.getElementById('inp-pregnancies').value) || 0,
        Smokes: parseInt(document.getElementById('inp-smokes').value, 10) || 0,
        'Smokes (years)': parseFloat(document.getElementById('inp-smokes-years').value) || 0,
        'Smokes (packs/year)': parseFloat(document.getElementById('inp-smokes-packs').value) || 0,
        'Hormonal Contraceptives': parseInt(document.getElementById('inp-hc').value, 10) || 0,
        'Hormonal Contraceptives (years)': parseFloat(document.getElementById('inp-hc-years').value) || 0,
        IUD: parseInt(document.getElementById('inp-iud').value, 10) || 0,
        'IUD (years)': parseFloat(document.getElementById('inp-iud-years').value) || 0,
        STDs: parseInt(document.getElementById('inp-stds').value, 10) || 0,
        'STDs:HPV': parseInt(document.getElementById('inp-std-hpv').value, 10) || 0,
        'STDs:HIV': parseInt(document.getElementById('inp-std-hiv').value, 10) || 0,
    };
}

function submitClinicalForm() {
    const data = getClinicalFormData();
    const btn = document.getElementById('btn-calculate-risk');
    if (btn) btn.innerHTML = 'Calculating...';

    fetch('/api/predict-clinical', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    })
    .then(res => res.json())
    .then(result => {
        if (btn) btn.innerHTML = '⚡ Calculate Clinical Risk & Suggestions';
        if (result.error) {
            alert('Clinical risk error: ' + result.error);
            return;
        }
        renderClinicalResult(result);
    })
    .catch(err => {
        if (btn) btn.innerHTML = '⚡ Calculate Clinical Risk & Suggestions';
        console.error('Clinical risk calculation error:', err);
    });
}

function handleCSVUpload(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict-clinical', {
        method: 'POST',
        body: formData,
    })
    .then(res => res.json())
    .then(result => {
        if (result.error) {
            alert('CSV upload error: ' + result.error);
            return;
        }
        renderClinicalResult(result);
        alert('Patient CSV clinical data parsed and evaluated successfully!');
    })
    .catch(err => {
        console.error('CSV upload error:', err);
        alert('Failed to upload clinical CSV.');
    });
}

function renderClinicalResult(result) {
    appState.latestClinicalResult = result;

    // 1. Risk Score & Gauge Dial
    const scoreValEl = document.getElementById('risk-score-val');
    const badgeEl = document.getElementById('clinical-risk-badge');
    const summaryEl = document.getElementById('clinical-summary-text');
    const dialCircle = document.getElementById('risk-dial-circle');

    const pct = result.risk_percentage || 0;
    if (scoreValEl) scoreValEl.textContent = `${pct.toFixed(1)}%`;

    // Circumference = 2 * PI * r = 2 * 3.14159 * 42 ≈ 264
    if (dialCircle) {
        const offset = 264 - (264 * (pct / 100));
        dialCircle.style.strokeDashoffset = offset;

        if (pct >= 50) {
            dialCircle.style.stroke = 'var(--rose-danger)';
        } else if (pct >= 22) {
            dialCircle.style.stroke = 'var(--amber-warning)';
        } else {
            dialCircle.style.stroke = 'var(--emerald-success)';
        }
    }

    if (badgeEl) {
        badgeEl.textContent = result.risk_level || 'Low Risk';
        badgeEl.className = 'stage-tag';
        if (result.risk_level === 'High Risk') {
            badgeEl.classList.add('stage-tag-critical');
        } else if (result.risk_level === 'Moderate Risk') {
            badgeEl.classList.add('stage-tag-warning');
        } else {
            badgeEl.classList.add('stage-tag-normal');
        }
    }

    if (summaryEl) summaryEl.textContent = result.summary_statement || '';

    // 2. Risk Drivers Breakdown
    const driversList = document.getElementById('risk-drivers-list');
    if (driversList) {
        driversList.innerHTML = '';
        if (result.key_risk_drivers && result.key_risk_drivers.length > 0) {
            result.key_risk_drivers.forEach(driver => {
                const card = document.createElement('div');
                card.className = 'driver-card';
                card.innerHTML = `
                    <div>
                        <div class="driver-title">${driver.label}</div>
                        <div class="driver-note">${driver.clinical_note}</div>
                    </div>
                    <div class="driver-weight">+${driver.importance_weight}%</div>
                `;
                driversList.appendChild(card);
            });
        } else {
            driversList.innerHTML = `
                <div style="font-size: 0.8rem; color: var(--emerald-success); padding: 0.5rem 0;">
                    ✓ No abnormal high-risk drivers detected for this patient.
                </div>
            `;
        }
    }

    // 3. Clinical Suggestions
    const suggestionsList = document.getElementById('suggestions-list');
    if (suggestionsList) {
        suggestionsList.innerHTML = '';
        if (result.clinical_suggestions && result.clinical_suggestions.length > 0) {
            result.clinical_suggestions.forEach(sug => {
                const card = document.createElement('div');
                card.className = 'suggestion-card';

                let prioClass = 'prio-routine';
                if (sug.priority === 'Urgent') prioClass = 'prio-urgent';
                else if (sug.priority === 'High') prioClass = 'prio-high';
                else if (sug.priority === 'Moderate') prioClass = 'prio-moderate';

                card.innerHTML = `
                    <div class="suggestion-header">
                        <span class="suggestion-title">${sug.title}</span>
                        <span class="suggestion-priority ${prioClass}">${sug.priority}</span>
                    </div>
                    <div class="suggestion-desc">${sug.detail}</div>
                `;
                suggestionsList.appendChild(card);
            });
        }
    }
}

// ==============================================================================
// TAB 3: GENERATE & DISPLAY COMBINED REPORT
// ==============================================================================
function generateAndShowReport() {
    if (!appState.latestImageResult) {
        alert('Please evaluate an image in Step 1 first.');
        switchTab('cytology');
        return;
    }
    if (!appState.latestClinicalResult) {
        submitClinicalForm();
    }

    const payload = {
        image_result: appState.latestImageResult,
        clinical_result: appState.latestClinicalResult,
        patient_info: {
            patient_id: 'PT-94821',
            evaluated_at: new Date().toLocaleDateString(),
        }
    };

    fetch('/api/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
    .then(res => res.json())
    .then(report => {
        if (report.error) {
            alert('Report error: ' + report.error);
            return;
        }
        renderReportView(report);
        switchTab('report');
    })
    .catch(err => {
        console.error('Report generation error:', err);
    });
}

function renderReportView(report) {
    const bannerTitle = document.getElementById('rep-triage-title');
    const urgencyVal = document.getElementById('rep-urgency-val');
    const bannerEl = document.getElementById('rep-triage-banner');

    if (bannerTitle) bannerTitle.textContent = report.overall_tier || '';
    if (urgencyVal) urgencyVal.textContent = report.urgency || '';
    if (bannerEl && report.tier_color) {
        bannerEl.style.borderColor = report.tier_color;
        bannerEl.style.background = `${report.tier_color}18`;
    }

    // Cytology block
    const cyto = report.cytology_summary;
    const thumbEl = document.getElementById('rep-cytology-thumb');
    const cellClassEl = document.getElementById('rep-cell-class');
    const aiConfEl = document.getElementById('rep-ai-conf');
    const bethesdaTagEl = document.getElementById('rep-bethesda-tag');
    const cytoDescEl = document.getElementById('rep-cytology-desc');

    if (thumbEl && appState.latestImageResult && appState.latestImageResult.visualizations) {
        thumbEl.src = appState.latestImageResult.visualizations.overlay;
    }
    if (cellClassEl) cellClassEl.textContent = cyto.cell_class || '';
    if (aiConfEl) aiConfEl.textContent = `${cyto.confidence.toFixed(2)}%`;
    if (bethesdaTagEl) bethesdaTagEl.textContent = `${cyto.bethesda_stage} (${cyto.cin_grade})`;
    if (cytoDescEl) cytoDescEl.textContent = report.clinical_summary || '';

    // Clinical block
    const patParams = appState.latestClinicalResult.patient_parameters || {};
    const ageEl = document.getElementById('rep-pat-age');
    const debutEl = document.getElementById('rep-pat-debut');
    const smokeEl = document.getElementById('rep-pat-smoking');
    const hcEl = document.getElementById('rep-pat-hc');
    const stdEl = document.getElementById('rep-pat-std');
    const riskEl = document.getElementById('rep-pat-risk');

    if (ageEl) ageEl.textContent = `${patParams.Age || 30} years`;
    if (debutEl) debutEl.textContent = `Age ${patParams['First sexual intercourse'] || 18}`;
    if (smokeEl) {
        const smk = patParams.Smokes > 0;
        smokeEl.textContent = smk ? `${patParams['Smokes (years)']} yrs (${patParams['Smokes (packs/year)']} pk/yr)` : 'Non-smoker';
    }
    if (hcEl) {
        const hcY = patParams['Hormonal Contraceptives (years)'] || 0;
        hcEl.textContent = hcY > 0 ? `${hcY} years` : 'No use';
    }
    if (stdEl) {
        const hasStd = (patParams.STDs > 0) || (patParams['STDs:HPV'] > 0);
        stdEl.textContent = hasStd ? 'Positive' : 'Negative';
        stdEl.style.color = hasStd ? 'var(--rose-danger)' : 'var(--emerald-success)';
    }
    if (riskEl) {
        riskEl.textContent = `${report.clinical_summary_details.risk_percentage.toFixed(1)}% (${report.clinical_summary_details.risk_level})`;
    }

    // Action plan
    const actionsContainer = document.getElementById('rep-actions-container');
    if (actionsContainer && report.action_items) {
        actionsContainer.innerHTML = '';
        report.action_items.forEach((act, idx) => {
            const row = document.createElement('div');
            row.style.cssText = 'padding: 0.6rem 0.8rem; background: rgba(255,255,255,0.02); border-radius: 6px; font-size: 0.82rem;';
            row.innerHTML = `
                <strong style="color: #fff;">${idx + 1}. ${act.title}</strong>
                <span style="font-size: 0.7rem; color: var(--cyan-accent); margin-left: 0.5rem; text-transform: uppercase;">[${act.priority}]</span>
                <div style="color: var(--text-secondary); margin-top: 0.2rem;">${act.detail}</div>
            `;
            actionsContainer.appendChild(row);
        });
    }
}
