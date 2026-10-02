"""PostgreSQL connection and schema setup for the FastAPI backend."""

import os
from pathlib import Path
from threading import Lock

import psycopg
from dotenv import load_dotenv
from fastapi import HTTPException, status
from psycopg.rows import dict_row


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
_database_ready = False
_database_lock = Lock()


def connect():
    if not DATABASE_URL:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "DATABASE_URL is not configured. Copy .env.example to .env and set it before using authentication.",
        )
    try:
        return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5)
    except psycopg.OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "PostgreSQL is unavailable. Start PostgreSQL and verify DATABASE_URL.",
        ) from exc


def initialize_database() -> None:
    global _database_ready
    if _database_ready:
        return
    with _database_lock:
        if _database_ready:
            return
        with connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    name VARCHAR(80) NOT NULL,
                    email VARCHAR(254) NOT NULL UNIQUE,
                    role VARCHAR(20) NOT NULL DEFAULT 'Researcher'
                        CHECK (role IN ('Researcher', 'Clinician')),
                    password_hash BYTEA NOT NULL,
                    salt BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'Researcher'")
            connection.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash CHAR(64) PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id BIGSERIAL PRIMARY KEY,
                    patient_code VARCHAR(40) NOT NULL UNIQUE,
                    full_name VARCHAR(120) NOT NULL,
                    age INTEGER CHECK (age IS NULL OR age BETWEEN 0 AND 120),
                    gender VARCHAR(40),
                    smoking_status VARCHAR(80),
                    alcohol_status VARCHAR(80),
                    hpv_status VARCHAR(80),
                    previous_screening TEXT,
                    family_history VARCHAR(80),
                    symptoms TEXT,
                    pregnancies INTEGER,
                    reproductive_health_info TEXT,
                    other_risk_factors TEXT,
                    notes TEXT,
                    created_by BIGINT REFERENCES users(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    prediction_key VARCHAR(80) UNIQUE,
                    patient_id BIGINT REFERENCES patients(id) ON DELETE SET NULL,
                    uploaded_by BIGINT REFERENCES users(id) ON DELETE SET NULL,
                    image_filename VARCHAR(255),
                    image_path TEXT,
                    selected_model VARCHAR(80) NOT NULL,
                    disease_prediction VARCHAR(120),
                    disease_confidence DOUBLE PRECISION,
                    early_risk_probability DOUBLE PRECISION,
                    risk_level VARCHAR(20),
                    resnet50_probability DOUBLE PRECISION,
                    efficientnet_b0_probability DOUBLE PRECISION,
                    ensemble_probability DOUBLE PRECISION,
                    gradcam_path TEXT,
                    model_version VARCHAR(120),
                    is_demo_prediction BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
                    action VARCHAR(120) NOT NULL,
                    table_name VARCHAR(120),
                    record_id BIGINT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id)")
        _database_ready = True


def database_available() -> bool:
    try:
        initialize_database()
        with connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return True
    except HTTPException:
        return False
