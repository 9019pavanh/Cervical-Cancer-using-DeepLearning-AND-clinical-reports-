"""
cancer_staging.py
=================
Comprehensive Cervical Cancer Staging & Classification Engine.

Maps cytology classifications (SIPaKMeD classes) to:
- The Bethesda System for Reporting Cervical Cytology (2014)
- Cervical Intraepithelial Neoplasia (CIN) histological grades
- International Federation of Gynecology and Obstetrics (FIGO) staging correlation
- Morphological hallmarks (Nuclear enlargement, Hyperchromasia, N:C ratio)
- Recommended clinical management (ASCCP consensus guidelines)
"""

from typing import Any, Dict


STAGING_DATABASE: Dict[str, Dict[str, Any]] = {
    "im_Dyskeratotic": {
        "class_id": 0,
        "name": "Dyskeratotic Epithelial Cells",
        "short_name": "Dyskeratotic",
        "bethesda_category": "HSIL (High-Grade Squamous Intraepithelial Lesion)",
        "cin_grade": "CIN 2 / CIN 3 (Moderate to Severe Dysplasia / Carcinoma In Situ)",
        "cancer_stage": "Stage 0 (Pre-Invasive High-Grade Cervical Neoplasia)",
        "stage_code": "HSIL / CIN 2-3",
        "severity": "Critical",
        "severity_color": "#ff3b30",
        "malignancy_risk": "High Oncogenic Potential (30-40% progress to invasive carcinoma if untreated)",
        "morphological_features": [
            "Marked nuclear enlargement with hyperchromasia",
            "Coarse, irregularly clumped chromatin distribution",
            "Markedly increased Nuclear-to-Cytoplasmic (N:C) ratio (> 0.7)",
            "Irregular nuclear membranes and prominent indentations",
            "Dense, orangeophilic or keratinized abnormal cytoplasm"
        ],
        "clinical_significance": (
            "Dyskeratotic cells signify severe squamous dysplasia or carcinoma in situ. "
            "These high-grade atypical cells represent true direct precursors of invasive squamous cell carcinoma."
        ),
        "immediate_action": "Urgent Colposcopy + Directed Biopsy with Endocervical Curettage (ECC) within 2-4 weeks.",
        "treatment_options": [
            "Loop Electrosurgical Excision Procedure (LEEP)",
            "Cold Knife Conization (CKC)",
            "Laser ablation / cryotherapy (if transformation zone fully visible)"
        ],
        "follow_up_protocol": "Post-treatment co-testing (cytology + hrHPV) at 6 and 12 months for 25 years."
    },
    "im_Koilocytotic": {
        "class_id": 1,
        "name": "Koilocytotic Squamous Cells",
        "short_name": "Koilocytotic",
        "bethesda_category": "LSIL (Low-Grade Squamous Intraepithelial Lesion / HPV Cytopathic Effect)",
        "cin_grade": "CIN 1 (Mild Dysplasia)",
        "cancer_stage": "Stage 0 (Early Pre-cancerous / Productive HPV Infection)",
        "stage_code": "LSIL / CIN 1",
        "severity": "Warning",
        "severity_color": "#ff9500",
        "malignancy_risk": "Moderate Risk (~60% regress spontaneously within 24 months, 10-15% progress to HSIL)",
        "morphological_features": [
            "Pathognomonic sharply demarcated perinuclear halo",
            "Nuclear enlargement (2-3x size of intermediate nucleus)",
            "Hyperchromatic, wrinkled nuclear contours ('raisinoid' nuclei)",
            "Bi-nucleation or multi-nucleation frequently present",
            "Condensed, thickened peripheral cytoplasmic rim"
        ],
        "clinical_significance": (
            "Koilocytes are the direct hallmark of productive human papillomavirus (HPV) infection. "
            "While most lesions are transient and cleared by host cell-mediated immunity, persistent hrHPV types require strict monitoring."
        ),
        "immediate_action": "Triage according to patient age and hrHPV status. Repeat co-testing in 12 months or colposcopy if hrHPV+.",
        "treatment_options": [
            "Expectant observation with serial co-testing (preferred in reproductive age)",
            "Colposcopic assessment if HPV-16/18 positive or persistent > 2 years"
        ],
        "follow_up_protocol": "Co-testing every 12 months until 2 consecutive negative results."
    },
    "im_Metaplastic": {
        "class_id": 2,
        "name": "Squamous Metaplastic Cells",
        "short_name": "Metaplastic",
        "bethesda_category": "Benign Reactive Cellular Changes / Squamous Metaplasia",
        "cin_grade": "Normal / Non-Dysplastic (Benign Transformation Zone)",
        "cancer_stage": "Stage 0 (Benign Physiological Transformation)",
        "stage_code": "Benign Metaplasia",
        "severity": "Informational",
        "severity_color": "#007aff",
        "malignancy_risk": "Very Low (< 2% risk; physiological transformation zone maturation)",
        "morphological_features": [
            "Polygonal or cobble-stone shaped cells with defined borders",
            "Centrally located, round to oval uniform nuclei",
            "Smooth nuclear membranes with fine, evenly distributed chromatin",
            "Moderate N:C ratio with dense, cyanophilic or amphophilic cytoplasm",
            "Intercellular bridges and distinct cytoplasmic projections ('spider cells')"
        ],
        "clinical_significance": (
            "Metaplasia is a normal physiological reparative mechanism where endocervical columnar epithelium "
            "transforms into protective squamous epithelium at the squamocolumnar junction. Immature metaplasia is more susceptible to HPV uptake."
        ),
        "immediate_action": "No immediate therapeutic intervention needed. Confirm absence of atypical nuclear features.",
        "treatment_options": [
            "Routine observation",
            "Treat concurrent cervical inflammation / vaginitis if symptomatic"
        ],
        "follow_up_protocol": "Standard population screening interval (every 3-5 years)."
    },
    "im_Parabasal": {
        "class_id": 3,
        "name": "Parabasal Epithelial Cells",
        "short_name": "Parabasal",
        "bethesda_category": "NILM (Negative for Intraepithelial Lesion or Malignancy) - Atrophic/Deep Layers",
        "cin_grade": "Normal Squamous Epithelium (Basal/Parabasal)",
        "cancer_stage": "Stage 0 (Normal / Non-Neoplastic)",
        "stage_code": "NILM (Parabasal)",
        "severity": "Success",
        "severity_color": "#34c759",
        "malignancy_risk": "Negative for Malignancy (< 0.5% risk)",
        "morphological_features": [
            "Small, rounded, elliptical cell morphology",
            "Centrally located round nuclei with smooth contour",
            "Regular, finely granular, vesicular chromatin pattern",
            "Dense, cyanophilic, well-defined cytoplasm",
            "Uniform cell size without pleomorphism"
        ],
        "clinical_significance": (
            "Parabasal cells originate from the deeper layers of the stratified squamous epithelium. "
            "Common in postmenopausal women (atrophic smear) or vigorous cervical sampling. Healthy non-cancerous finding."
        ),
        "immediate_action": "Standard negative reporting. Correlate with hormonal/menopausal status.",
        "treatment_options": [
            "None required for non-neoplastic finding",
            "Topical estrogen therapy if atrophic vaginitis causes clinical symptoms"
        ],
        "follow_up_protocol": "Routine cervical cancer screening interval per national guidelines."
    },
    "im_Superficial-Intermediate": {
        "class_id": 4,
        "name": "Superficial & Intermediate Squamous Cells",
        "short_name": "Superficial-Intermediate",
        "bethesda_category": "NILM (Negative for Intraepithelial Lesion or Malignancy) - Mature Squamous",
        "cin_grade": "Normal Mature Squamous Epithelium",
        "cancer_stage": "Stage 0 (Normal / Healthy Mature Cells)",
        "stage_code": "NILM (Mature)",
        "severity": "Success",
        "severity_color": "#30d158",
        "malignancy_risk": "Negative for Malignancy (< 0.2% risk)",
        "morphological_features": [
            "Large, polygonal, flat squamous cells with abundant transparent cytoplasm",
            "Small, pyknotic or vesicular central nucleus with low N:C ratio (< 0.15)",
            "Delicate, translucent cytoplasm (pink/eosinophilic for superficial, blue-green for intermediate)",
            "Smooth, regular, round-to-oval nuclear membranes",
            "Absence of hyperchromasia or nucleoli"
        ],
        "clinical_significance": (
            "Mature, fully differentiated squamous cells from the surface of the normal ectocervix. "
            "Reflects optimal estrogenic and progestational maturation. Ideal healthy cytology finding."
        ),
        "immediate_action": "Standard normal reporting. Reassure patient of negative test result.",
        "treatment_options": [
            "None (Healthy Normal Epithelium)"
        ],
        "follow_up_protocol": "Standard routine screening interval (co-testing every 5 years or cytology every 3 years)."
    }
}


def get_staging_info(class_name: str) -> Dict[str, Any]:
    """Retrieves full staging metadata for a given class name."""
    clean_name = class_name.strip()
    if clean_name in STAGING_DATABASE:
        return STAGING_DATABASE[clean_name]

    # Search by partial match
    for key, data in STAGING_DATABASE.items():
        if key.lower() in clean_name.lower() or data["short_name"].lower() in clean_name.lower():
            return data

    # Default fallback
    return {
        "class_id": -1,
        "name": class_name,
        "short_name": class_name,
        "bethesda_category": "Indeterminate Cytology",
        "cin_grade": "Pending Histopathology",
        "cancer_stage": "Unclassified",
        "stage_code": "Pending",
        "severity": "Informational",
        "severity_color": "#8e8e93",
        "malignancy_risk": "Requires Pathologist Review",
        "morphological_features": ["Morphology requires visual confirmation"],
        "clinical_significance": "Atypical cytology requiring diagnostic review.",
        "immediate_action": "Review by board-certified cytopathologist.",
        "treatment_options": ["Consultation and repeat examination."],
        "follow_up_protocol": "Clinical follow-up within 3 months."
    }
