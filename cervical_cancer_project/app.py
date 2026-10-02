"""FastAPI server for the installable research dashboard."""

import io
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from auth import (
    authenticate_user,
    create_session,
    current_user,
    delete_session,
    register_user,
    require_user,
)
from dataset import audit_project
from database import database_available
from demo_data import DEMO_DISCLAIMER, DEMO_PATIENTS
from inference import ResearchInference


ROOT = Path(__file__).resolve().parent
PWA_DIR = ROOT / "app" / "pwa"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

api = FastAPI(
    title="Explainable Cervical AI Research Framework",
    version="1.0.0",
    description="Research-only cervical cytology and early-risk prediction API.",
)
inference = ResearchInference()
api.mount("/assets", StaticFiles(directory=PWA_DIR), name="assets")


@api.middleware("http")
async def protect_browser_responses(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path not in {"/docs", "/redoc", "/openapi.json"}:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
    if request.url.path.startswith("/api/") or request.url.path in {"/dashboard", "/login", "/register"}:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@api.get("/")
def home():
    return FileResponse(PWA_DIR / "landing.html")


@api.get("/preview")
def preview_landing():
    return FileResponse(PWA_DIR / "landing.html")


@api.get("/preview/login")
@api.get("/preview/register")
def preview_authentication():
    return FileResponse(PWA_DIR / "auth.html")


@api.get("/preview/dashboard")
def preview_dashboard():
    return FileResponse(PWA_DIR / "index.html")


@api.get("/login")
@api.get("/register")
def authentication_page(request: Request):
    try:
        user = current_user(request)
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        user = None
    if user:
        return RedirectResponse("/dashboard", status_code=303)
    return FileResponse(PWA_DIR / "auth.html")


@api.get("/dashboard")
def dashboard(request: Request):
    if current_user(request) is None:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(PWA_DIR / "index.html")


@api.get("/manifest.webmanifest")
def manifest():
    return FileResponse(PWA_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@api.get("/service-worker.js")
def service_worker():
    return FileResponse(
        PWA_DIR / "service-worker.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@api.get("/api/status")
def status(user: dict = Depends(require_user)):
    return inference.status()


@api.get("/api/health")
def health():
    return {
        "api": "ready",
        "postgresql": "ready" if database_available() else "unavailable",
        "image_models": sorted(inference.models),
    }


@api.get("/api/demo/patients")
def demo_patients():
    return {"is_demo_dataset": True, "disclaimer": DEMO_DISCLAIMER, "patients": DEMO_PATIENTS}


@api.get("/api/audit")
def audit(user: dict = Depends(require_user)):
    report = audit_project(write_report=True)
    return report.__dict__


@api.post("/api/image")
async def predict_image(
    model: str = Form(...),
    image: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit")
    try:
        with Image.open(io.BytesIO(content)) as uploaded:
            if uploaded.width * uploaded.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image dimensions are too large")
            uploaded.load()
            pil_image = uploaded.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Invalid image file") from exc
    try:
        return await run_in_threadpool(inference.predict_image, pil_image, model)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc


@api.post("/api/clinical")
def predict_clinical(values: dict, user: dict = Depends(require_user)):
    try:
        return inference.predict_clinical(values)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc


class RegistrationPayload(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=10, max_length=128)
    role: str = Field(default="Researcher", min_length=1, max_length=20)


class LoginPayload(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


@api.post("/api/auth/register")
def register(payload: RegistrationPayload, request: Request, response: Response):
    user = register_user(payload.name, payload.email, payload.password, payload.role)
    create_session(response, user["id"], secure=request.url.scheme == "https")
    return {"user": user}


@api.post("/api/auth/login")
def login(payload: LoginPayload, request: Request, response: Response):
    user = authenticate_user(payload.email, payload.password)
    if user is None:
        raise HTTPException(401, "Invalid email or password")
    create_session(response, user["id"], secure=request.url.scheme == "https")
    return {"user": user}


@api.post("/api/auth/logout")
def logout(request: Request, response: Response):
    delete_session(request, response)
    return {"ok": True}


@api.get("/api/auth/me")
def me(user: dict = Depends(require_user)):
    return {"user": user}


app = api
