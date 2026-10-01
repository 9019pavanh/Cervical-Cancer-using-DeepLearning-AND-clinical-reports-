# Deployment and GitHub

## Local application

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The model directory defaults to `artifacts/models`. Override `MODEL_DIR` to mount another location. Startup loads checkpoints once; restart after replacing weights. Inference is serialized inside a process to keep Grad-CAM hooks independent. Use one worker unless memory has been measured; each worker loads both models. The inference time reported in the interface includes explanations, but excludes upload and HTTP overhead.

## Docker on a server

Provision a Linux server with Docker, at least 4 GB RAM (8 GB recommended), and storage for dependencies and weights. Transfer `resnet50.pt`, `efficientnet_b0.pt`, and both `*-metrics.json` reports into `artifacts/models/` via your private artifact storage or secure copy. Do not transfer manifests containing local paths or image datasets.

```sh
docker compose up --build -d
curl http://localhost:8000/api/health
```

The service can be healthy while the models are absent; verify that `ready` is `true` and that the desired task is listed. Put an HTTPS reverse proxy in front of port 8000. For publicly accessible deployments, add authentication and rate limits at the proxy and cap request bodies at 10 MB plus multipart overhead. The application limits image bytes and pixel dimensions, processes images in memory, and writes no upload files. Do not send identifiable clinical material to a public research demo.

## GitHub

Target repository: https://github.com/9019pavanh/Cervical-Cancer-using-DeepLearning-AND-clinical-reports-

This update preserves upstream files and prepares branch `feature/resnet50-efficientnet-research`. Authenticate GitHub locally if publishing is unavailable:

```powershell
gh auth login
git push -u origin feature/resnet50-efficientnet-research
```

Open a pull request into `main`, review CI, and merge when ready. GitHub stores source code; running inference also needs compute and trained weights. GitHub Pages cannot run this Python model server. No hosted deployment URL is claimed until a hosting account is configured and readiness is checked there.
