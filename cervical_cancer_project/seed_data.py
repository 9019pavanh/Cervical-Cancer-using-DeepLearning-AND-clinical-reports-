"""Seed the PostgreSQL database with clearly synthetic demo patients and predictions."""

from datetime import datetime, timezone

from database import connect, initialize_database
from demo_data import DEMO_PATIENTS


def seed_demo_data() -> int:
    initialize_database()
    inserted = 0
    with connect() as connection:
        for index, demo in enumerate(DEMO_PATIENTS, start=1):
            patient_code = f"DEMO-{index:03d}"
            patient = connection.execute(
                """
                INSERT INTO patients(
                    patient_code,full_name,age,gender,smoking_status,alcohol_status,
                    hpv_status,previous_screening,family_history,symptoms,notes
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (patient_code) DO UPDATE SET
                    full_name=EXCLUDED.full_name, age=EXCLUDED.age, gender=EXCLUDED.gender,
                    smoking_status=EXCLUDED.smoking_status, alcohol_status=EXCLUDED.alcohol_status,
                    hpv_status=EXCLUDED.hpv_status, previous_screening=EXCLUDED.previous_screening,
                    family_history=EXCLUDED.family_history, symptoms=EXCLUDED.symptoms,
                    notes=EXCLUDED.notes, updated_at=NOW()
                RETURNING id
                """,
                (
                    patient_code,
                    demo["patient_name"],
                    demo["age"],
                    demo["gender"],
                    "Yes" if any(
                        "current smoker" in factor.lower() or "former smoker" in factor.lower()
                        for factor in demo["clinical_risk_factors"]
                    ) else "No",
                    "Synthetic demo record",
                    next((factor for factor in demo["clinical_risk_factors"] if "HPV" in factor), "Unknown"),
                    demo["screening_history"],
                    "Yes" if any("family history" in factor.lower() for factor in demo["clinical_risk_factors"]) else "No",
                    demo["symptoms"],
                    demo["disclaimer"],
                ),
            ).fetchone()
            patient_id = patient["id"]
            connection.execute(
                """
                INSERT INTO predictions(
                    prediction_key,patient_id,image_filename,image_path,selected_model,
                    disease_prediction,disease_confidence,early_risk_probability,risk_level,
                    resnet50_probability,efficientnet_b0_probability,ensemble_probability,
                    gradcam_path,model_version,is_demo_prediction,created_at
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (prediction_key) DO UPDATE SET
                    patient_id=EXCLUDED.patient_id, selected_model=EXCLUDED.selected_model,
                    early_risk_probability=EXCLUDED.early_risk_probability, risk_level=EXCLUDED.risk_level,
                    resnet50_probability=EXCLUDED.resnet50_probability,
                    efficientnet_b0_probability=EXCLUDED.efficientnet_b0_probability,
                    ensemble_probability=EXCLUDED.ensemble_probability,
                    gradcam_path=EXCLUDED.gradcam_path, is_demo_prediction=TRUE
                """,
                (
                    f"demo-prediction-{index:03d}",
                    patient_id,
                    "synthetic-demo.svg",
                    demo["gradcam_path"],
                    demo["selected_model"],
                    demo["predicted_disease"],
                    demo["disease_probability"],
                    demo["early_risk_probability"],
                    demo["risk_category"],
                    demo["resnet50_confidence"],
                    demo["efficientnet_b0_confidence"],
                    demo["ensemble_confidence"],
                    demo["gradcam_path"],
                    "synthetic-demo-v1",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            inserted += 1
    return inserted


if __name__ == "__main__":
    print(f"Seeded {seed_demo_data()} synthetic demo patients.")
