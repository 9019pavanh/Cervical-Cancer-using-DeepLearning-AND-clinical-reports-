import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .service import Predictor, decode_image

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor(os.getenv('MODEL_DIR', str(ROOT/'artifacts'/'models')))
    yield


app = FastAPI(title='CerviSight Research API', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT/'frontend'), name='static')


@app.middleware('http')
async def reject_large_requests(request, call_next):
    from fastapi.responses import JSONResponse
    length = request.headers.get('content-length')
    if request.url.path == '/api/predict' and length:
        try:
            if int(length) > MAX_BYTES + 64 * 1024:
                return JSONResponse({'detail':'Request exceeds upload limit'},status_code=413)
        except ValueError:
            return JSONResponse({'detail':'Invalid content length'},status_code=400)
    return await call_next(request)


@app.get('/')
def home():
    return FileResponse(ROOT/'frontend'/'index.html')


@app.get('/api/health')
def health():
    return {'service':'ok', **app.state.predictor.status()}


@app.get('/api/metrics')
def metrics():
    return app.state.predictor.metrics()


@app.post('/api/predict')
async def predict(file: UploadFile = File(...), task: str = Form('bethesda')):
    if not app.state.predictor.status()['ready']:
        raise HTTPException(503, 'Trained ResNet50 and EfficientNet checkpoints are required.')
    content = await file.read(MAX_BYTES+1)
    await file.close()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, 'Image exceeds 10 MB')
    try:
        image = decode_image(content)
    except Exception:
        raise HTTPException(400, 'Invalid image. Use JPEG, PNG, BMP, or TIFF up to 25 megapixels.')
    try:
        return await run_in_threadpool(app.state.predictor.predict, image, task)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
