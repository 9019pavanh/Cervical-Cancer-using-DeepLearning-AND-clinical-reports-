# An Explainable Multimodal Hybrid Deep Learning Framework for Cervical Disease Classification and Early Cervical Cancer Risk Prediction

This repository is a reproducible **research prototype** for cervical-cell image classification and independent early-risk modelling. It is not a medical device and must not be used as a diagnosis or treatment recommendation.

## Dataset audit

The included audit found:

- 5,015 readable SIPaKMeD images and no corrupt files.
- No exact duplicate image files.
- Five classes:
  - `im_Dyskeratotic` — 1,036 images
  - `im_Koilocytotic` — 1,063 images
  - `im_Metaplastic` — 1,064 images
  - `im_Parabasal` — 895 images
  - `im_Superficial-Intermediate` — 957 images
- 858 clinical rows with 36 columns in the UCI cervical-risk CSV.
- No patient identifier shared by the image and clinical datasets.

The SIPaKMeD numeric filename prefix is treated as a **source group**. An original image such as `032.bmp` and crops such as `032_01.bmp` are always assigned to the same split.

## Scientific limitation

Feature-level multimodal training is not valid with the available files because no verified key connects a microscopy image to its patient's clinical record. The code includes the requested CNN + clinical MLP fusion architecture, but `train.py --task multimodal` deliberately refuses to train until a linkage table exists.

The valid experiments are:

1. ResNet50 image classification.
2. EfficientNet-B0 image classification.
3. Independent clinical early-risk prediction using pre-diagnostic factors.

Do not randomly pair clinical rows with images. Such pairing creates a visually impressive but scientifically meaningless “hybrid” model.

## Architecture

### Image branch

- ImageNet-pretrained ResNet50 or EfficientNet-B0.
- Final classifier replaced for the five discovered classes.
- 224 × 224 input, ImageNet normalization, training-only augmentation.
- AdamW, cosine learning-rate scheduling, class-weighted focal loss, label smoothing, early stopping on validation macro-F1.

### Clinical branch

- Missing-value imputation fitted only on training data.
- Numerical standardization and categorical one-hot encoding.
- Diagnostic-result and target columns removed from input features.
- MLP with batch normalization, GELU and dropout.
- Binary biopsy risk output evaluated using ROC-AUC, PR-AUC, calibration and Brier score.

### Guarded multimodal branch

- CNN image embedding concatenated with clinical MLP embedding.
- Batch normalization, dropout and fully connected fusion layers.
- Disease-classification and risk-probability heads.
- Enabled only after a verified `image_path ↔ patient_id ↔ clinical row` mapping is supplied.

## Leakage prevention

- Related image crops are grouped before train/validation/test splitting.
- Exact duplicates and unreadable files are excluded by the audit.
- Augmentation is applied only to training images.
- Clinical preprocessing is fitted only on the clinical training partition.
- `Biopsy` is the clinical target and never an input.
- `Hinselmann`, `Schiller`, `Citology`, and diagnosis columns are excluded as outcome/proxy leakage.
- Validation selects checkpoints; the test split is used only for final evaluation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### PostgreSQL

Authentication uses PostgreSQL; no SQLite database is used. Start a local PostgreSQL instance with Docker:

```powershell
Copy-Item .env.example .env
$env:POSTGRES_PASSWORD = "cervical_ai_dev"  # local development only
$env:DATABASE_URL = "postgresql://cervical_ai:cervical_ai_dev@127.0.0.1:5432/cervical_ai"
docker compose up -d database
```

If Docker is not installed on Windows, install PostgreSQL 17 once with `winget install --id PostgreSQL.PostgreSQL.17`, choose a local development password during the installer, and set the matching `DATABASE_URL` in `.env`. The project cannot create a PostgreSQL server itself; it creates the application tables after the server is available.

For a deployed environment, set `DATABASE_URL` to a private PostgreSQL connection string and do not commit `.env`.

Initialize the application schema and insert the five synthetic backend records after PostgreSQL is running:

```powershell
\.\.venv\Scripts\python.exe seed_data.py
```

## Audit and create grouped splits

```powershell
.\.venv\Scripts\python.exe dataset.py
```

The report is saved as `results/research/dataset_audit.json`.

## Train image models

```powershell
.\.venv\Scripts\python.exe train.py --task image --model resnet50 --epochs 30
.\.venv\Scripts\python.exe train.py --task image --model efficientnet_b0 --epochs 30
```

For a CPU-only machine, training can take several hours. CUDA is selected automatically when available.

## Train the independent clinical model

```powershell
.\.venv\Scripts\python.exe train.py --task clinical --epochs 50
```

## Final evaluation

Run this only after model selection is complete:

```powershell
.\.venv\Scripts\python.exe evaluate.py --task image --model resnet50
.\.venv\Scripts\python.exe evaluate.py --task image --model efficientnet_b0
.\.venv\Scripts\python.exe evaluate.py --task clinical
```

Outputs include JSON metrics, CSV predictions, confusion matrices, ROC curves, precision-recall curves, calibration plots and 95% bootstrap confidence intervals.

Never report validation accuracy as final accuracy. A target above 97% should be reported only when produced by the untouched grouped test split.

## Explainability

```powershell
.\.venv\Scripts\python.exe explainability.py --model resnet50 --image path\to\image.bmp
.\.venv\Scripts\python.exe explainability.py --model efficientnet_b0 --image path\to\image.bmp
.\.venv\Scripts\python.exe explainability.py --clinical
```

## Run the mobile-ready application

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`. On Android Chrome or Microsoft Edge, choose **Install app** or **Add to Home screen**. Installation requires HTTPS when accessed from another device; localhost is allowed during development.

Public visual previews are available without PostgreSQL at `/preview`, `/preview/login`, `/preview/register`, and `/preview/dashboard`. Preview forms do not submit and every patient card is synthetic.

The PWA caches only the application shell. Model inference still requires the FastAPI server because PyTorch checkpoints run on the server.

### Login and registration

- New users register with name, email and a password containing at least 10 characters, one letter and one number.
- Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes with 600,000 iterations.
- Random session tokens are stored only as SHA-256 hashes and expire after seven days.
- Authentication uses an HttpOnly, SameSite cookie; HTTPS deployments also receive the `Secure` cookie flag.
- Authenticated pages and API responses use `no-store` caching; the service worker caches only public static assets.
- Same-origin checks and browser security headers protect the dashboard endpoints.
- Dashboard and prediction endpoints require authentication.
- Uploaded images and clinical form values are processed in memory and are not persisted.
- PostgreSQL credentials are read from `DATABASE_URL` and never sent to the browser.

### Synthetic demo gallery

The dashboard includes five clearly labelled synthetic demonstrations: Ananya Rao, Kavya Sharma, Priya Nair, Meera Joshi, and Sneha Reddy. Each card separates patient information, image classification, clinical-risk estimate, Grad-CAM illustration, and final research interpretation. The dataset contains cell classes only, so the cards use **Predicted Cell Class** and do not invent disease diagnoses.

The demo payload is served by `GET /api/demo/patients` and includes `is_demo_prediction: true`, the exact requested confidence fields, risk category, clinical factors, result summary, and a synthetic Grad-CAM path.

## Current measured artifacts

The standardized EfficientNet-B0 checkpoint has been evaluated on 748 untouched grouped test images:

- Accuracy: **93.58%** (95% bootstrap CI: **91.71–95.32%**)
- Macro precision: **94.03%**
- Macro sensitivity/recall: **94.11%**
- Macro F1: **93.97%**
- Macro ROC-AUC: **0.9934**
- Macro PR-AUC: **0.9794**
- Multiclass Brier score: **0.1071**

The independent clinical MLP achieved ROC-AUC **0.6395**, PR-AUC **0.1545**, sensitivity **0.625**, and Brier score **0.2530** on its untouched test split. This is modest performance and must be treated as exploratory.

ResNet50 metrics are intentionally left blank until ResNet50 is trained and evaluated with the commands above. The earlier ResNet18 result is not relabelled as ResNet50.

## Project files

- `config.py` — paths and reproducible defaults
- `dataset.py` — audit, deduplication and grouped splitting
- `preprocessing.py` — image and clinical preprocessing
- `models.py` — ResNet50, EfficientNet-B0, clinical MLP and guarded hybrid
- `train.py` — image and clinical training
- `evaluate.py` — final metrics and result artifacts
- `explainability.py` — Grad-CAM and clinical importance
- `inference.py` — checkpoint loading and inference
- `app.py` — FastAPI application
- `auth.py` — PostgreSQL authentication and session management
- `database.py` — PostgreSQL connection, schema, indexes, and health check
- `seed_data.py` — idempotent synthetic patient/prediction seed
- `demo_data.py` — five synthetic preview records
- `docker-compose.yml` — local PostgreSQL service
- `app/pwa/landing.html` — public dark-blue product overview
- `app/pwa/auth.html` — login and registration pages
- `app/pwa/index.html` — responsive research dashboard and demo gallery
- `requirements.txt` — dependencies
- `results/research/EVALUATION_REPORT.md` — measured results and scientific limitations

## Required data for valid multimodal fusion

To enable true multimodal training, add a de-identified linkage file with at least:

```text
image_path,patient_id
.../cell_001.bmp,P0001
```

The clinical table must contain the same `patient_id`. Split by patient before preprocessing or model training.
