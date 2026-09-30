"""
server.py
=========
Flask Diagnostic Web Application Server for Cervical Cancer Screening & Staging.

Provides:
- Multi-model Cytology Analysis (ResNet-18, EfficientNet-B0, Dual Hybrid Fusion)
- Explainable AI Grad-CAM Visual Saliency Maps
- Cervical Cancer Staging (Bethesda System 2014 & CIN Grades)
- Patient Clinical Risk Assessment (UCI Risk Factors ML Engine)
- Unified Multimodal Diagnostic Report Synthesis
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# Add project root and src/ to Python path
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.cancer_staging import STAGING_DATABASE, get_staging_info
from src.hybrid_model import MODEL_BENCHMARKS
from src.inference_service import CLASS_NAMES, DiagnosticService

app = Flask(__name__, static_folder=str(SERVER_DIR / "static"), static_url_path="")

# Initialize diagnostic service singleton
print("[Server] Initializing DiagnosticService...")
diagnostic_service = DiagnosticService()
print("[Server] DiagnosticService ready.")


# ==============================================================================
# STATIC & WEB UI ROUTES
# ==============================================================================
@app.route("/")
def index():
    """Serves the main diagnostic single-page application."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serves static assets."""
    return send_from_directory(app.static_folder, filename)


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check and loaded model status."""
    return jsonify({
        "status": "healthy",
        "service": "Cervical Cancer AI Diagnostic Suite",
        "device": str(diagnostic_service.device),
        "models_loaded": {
            "resnet18": diagnostic_service.resnet_model is not None,
            "hybrid_model": diagnostic_service.hybrid_model is not None,
            "clinical_predictor": diagnostic_service.clinical_predictor is not None,
        },
        "version": "2.0.0"
    })


@app.route("/api/models-info", methods=["GET"])
def get_models_info():
    """Returns benchmark metrics, architectures, and capabilities of all models."""
    return jsonify({
        "benchmarks": MODEL_BENCHMARKS,
        "classes": [
            {
                "id": i,
                "key": cls,
                "name": STAGING_DATABASE[cls]["name"],
                "short_name": STAGING_DATABASE[cls]["short_name"],
                "bethesda": STAGING_DATABASE[cls]["bethesda_category"],
                "stage": STAGING_DATABASE[cls]["cancer_stage"],
                "color": STAGING_DATABASE[cls]["severity_color"],
            }
            for i, cls in enumerate(CLASS_NAMES)
        ]
    })


@app.route("/api/samples", methods=["GET"])
def get_sample_images():
    """Returns list of curated test cytology samples available for one-click testing."""
    samples_dir = SERVER_DIR / "static" / "samples"
    samples = []

    for cls in CLASS_NAMES:
        png_name = f"{cls}.png"
        sample_path = samples_dir / png_name
        staging = STAGING_DATABASE.get(cls, {})

        if sample_path.exists():
            samples.append({
                "sample_id": cls,
                "class_name": cls,
                "short_name": staging.get("short_name", cls),
                "bethesda_stage": staging.get("stage_code", "NILM"),
                "severity_color": staging.get("severity_color", "#007aff"),
                "thumbnail_url": f"/samples/{png_name}",
            })

    return jsonify({"samples": samples})


@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    """
    Accepts uploaded image file OR JSON sample_id/base64,
    runs multi-model inference, Grad-CAM, staging, and accuracy comparison.
    """
    try:
        image_input = None

        # Check if file was uploaded
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected for upload"}), 400
            image_input = io.BytesIO(file.read())

        # Check if JSON payload specified a sample or base64
        elif request.is_json:
            data = request.get_json()
            if "sample_id" in data:
                sample_path = SERVER_DIR / "static" / "samples" / f"{data['sample_id']}.png"
                if not sample_path.exists():
                    return jsonify({"error": f"Sample image '{data['sample_id']}' not found"}), 404
                image_input = sample_path
            elif "image_base64" in data:
                import base64
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                image_input = io.BytesIO(base64.b64decode(b64_str))

        if image_input is None:
            return jsonify({"error": "No image provided. Upload a file or specify a sample_id."}), 400

        # Execute comprehensive analysis
        analysis_result = diagnostic_service.analyze_cytology_image(image_input)
        return jsonify(analysis_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict-clinical", methods=["POST"])
def predict_clinical():
    """
    Evaluates patient clinical risk factors.
    Supports either JSON form values or uploaded patient CSV.
    """
    try:
        patient_data: Dict[str, Any] = {}

        # 1. Check if CSV file was uploaded
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                df_uploaded = pd.read_csv(io.BytesIO(file.read()))
                df_uploaded = df_uploaded.replace("?", None)
                if len(df_uploaded) > 0:
                    patient_data = df_uploaded.iloc[0].to_dict()
                else:
                    return jsonify({"error": "Uploaded CSV contains no rows."}), 400

        # 2. Or parse JSON form data
        elif request.is_json:
            patient_data = request.get_json() or {}

        if not patient_data:
            return jsonify({"error": "No clinical data provided."}), 400

        # Run clinical risk prediction
        clinical_result = diagnostic_service.assess_clinical_risk(patient_data)
        return jsonify(clinical_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    """
    Combines cytology image results + clinical risk assessment
    into an integrated multimodal diagnostic report.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "JSON payload required."}), 400

        payload = request.get_json()
        image_result = payload.get("image_result")
        clinical_result = payload.get("clinical_result")
        patient_info = payload.get("patient_info", {})

        if not image_result or not clinical_result:
            return jsonify({"error": "Both image_result and clinical_result are required to generate the synthesized report."}), 400

        report = diagnostic_service.generate_multimodal_synthesis(
            image_result=image_result,
            clinical_result=clinical_result,
            patient_metadata=patient_info,
        )
        return jsonify(report)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  CERVICAL CANCER DIAGNOSTIC WEB APPLICATION")
    print(f"  Live local server: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
