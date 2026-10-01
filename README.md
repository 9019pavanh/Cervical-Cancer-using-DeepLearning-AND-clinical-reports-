# CerviSight

**An Explainable Multimodal Hybrid Deep Learning Framework for Cervical Disease Classification and Early Cervical Cancer Risk Prediction**

Current implementation: an explainable **image-only research baseline** using pretrained ResNet50 and EfficientNet-B0 with independently trained cytology classification heads and equal-weight probability fusion. The proposed title describes the broader research objective; patient-linked clinical fusion and future cancer-risk prediction are not implemented or validated. Suitable current implementation title: **An Explainable ResNet50–EfficientNet Ensemble for Cervical Cytology Image Classification**.

## Run on Windows

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Copy configs/datasets.example.json to configs/datasets.local.json and set your dataset paths.
.\.venv\Scripts\python.exe -m ml.catalog
.\.venv\Scripts\python.exe -m ml.train
.\.venv\Scripts\python.exe -m ml.evaluate
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. The application includes image upload, taxonomy selection, ensemble and individual model scores, Grad-CAM overlays, downloadable JSON reports, and held-out evaluation. Check `/api/health` for model readiness. Without both trained checkpoints, prediction returns HTTP 503; no fabricated predictions are produced.

For a deliberately limited pilot, use `python -m ml.train --max-per-class 20`. This caps each class **in each split**, writes `pilot` metadata, and is unsuitable for accuracy claims. Omit this argument to use all admitted images. Default training extracts frozen ImageNet features and trains weighted task heads for 100 epochs, selecting the head using validation loss. This is transfer learning of classification heads, not full backbone fine-tuning. Internet access is required for official ImageNet weights on first training. Models and manifests remain in `artifacts/` and are excluded from Git.

## Data and evaluation

Five supplied directories are audited without modifying originals. Bethesda labels from Mendeley and Cytolog are harmonized conservatively; SIPaKMeD, Herlev, and custom binary labels retain separate heads. SIPaKMeD uses isolated `CROPPED` cells rather than mixing cells with whole fields. An identical RGB pixel image is deduplicated across all datasets; contradictory task/label copies are excluded. Duplicate groups are joined with inferred slide groups before a seeded 70/15/15 group split. The `CervicalCancer` directory repeats other collections, so auditing all directories is essential.

`artifacts/audit.json` records counts and exclusions; `artifacts/manifest.json` records original paths, labels, pixel hashes and splits. Model reports contain per-class precision/recall/F1, balanced accuracy, confusion matrices, split sizes, loss histories, and the source-manifest hash. Model scores are uncalibrated. An equal-weight probability ensemble combines the two architectures; Grad-CAM overlays average their importance maps for the ensemble-selected class.

Filename groups are **slide proxies, not verified patient identities**. Pixel hashing does not catch resized, recompressed, or transformed duplicates. A trusted patient/slide registry and external institution test set are needed before publishing generalization claims. The existing repository's `cervical_cancer_project/` and clinical files are retained as legacy material; the new app does not execute that server or its clinical model. Its unpaired clinical CSV cannot establish an image–clinical multimodal cohort or future-risk endpoint.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md). A Docker image serves both the API and frontend. Model weights must be provisioned separately into a read-only model directory. CI runs meaningful pipeline and API tests without downloading pretrained weights.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
docker compose up --build -d
```

## Clinical fusion research extension

To implement the full proposed title, collect a consented, patient-linked cohort with image IDs, patient/slide IDs, clinical variables, and reference outcomes. Define whether the endpoint is current biopsy-confirmed disease or future cancer within a specified time horizon; these are different prediction problems. Use patient-level splits, fit imputers/scalers on training patients only, encode missing clinical values, and fuse a clinical MLP embedding with image embeddings. Compare image-only, clinical-only and fusion models, calibrate on validation data, and evaluate on external patients with confidence intervals and appropriate subgroup analyses. Until then the API explicitly returns `cancer_risk: null` and `clinical_fusion_used: false`.

Research demonstration only. Neither cytology morphology classes nor Grad-CAM maps establish cancer stage. No patient-care validation or clinical approval is claimed. Uploaded images are not persisted by the new server. Dataset licensing and redistribution rights must be checked before sharing data or weights. Existing legacy dataset files were already present upstream; no downloaded image datasets are added by this update.
