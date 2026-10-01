import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from ml.catalog import TASKS
from ml.models import CytologyModel, gradcam, preprocessing


class Predictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.models = []
        self.lock = threading.Lock()
        torch.set_num_threads(4)
        for architecture in ['resnet50', 'efficientnet_b0']:
            checkpoint = self.model_dir/f'{architecture}.pt'
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['metadata']['architecture'] != architecture or saved['metadata']['classes'] != TASKS:
                    raise ValueError(f'Incompatible checkpoint metadata: {architecture}')
                model = CytologyModel(architecture)
                model.load_state_dict(saved['state_dict'])
                model.eval()
                self.models.append((model, saved['metadata']))
        if len(self.models) == 2 and self.models[0][1]['manifest_sha256'] != self.models[1][1]['manifest_sha256']:
            raise ValueError('Ensemble checkpoints were trained using different manifests.')

    def status(self):
        return {'ready': len(self.models) == 2,
                'models': [metadata for _, metadata in self.models],
                'tasks': [task for task in TASKS if len(self.models) == 2 and all(task in meta['tasks'] for _,meta in self.models)],
                'clinical_fusion_available': False, 'cancer_risk_prediction_available': False,
                'intended_use': 'Research demonstration; not validated for patient care.'}

    def metrics(self):
        return {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in self.model_dir.glob('*-metrics.json')}

    def predict(self, image, task):
        if task not in self.status()['tasks']:
            raise ValueError('Both trained models must support the selected task.')
        start = time.perf_counter()
        tensor = preprocessing()(image)[None]
        with self.lock:
            with torch.no_grad():
                individual = [model(tensor,task).softmax(1)[0].numpy() for model,_ in self.models]
            probabilities = np.mean(individual,axis=0)
            index = int(probabilities.argmax())
            cams = [gradcam(model,tensor,task,index) for model,_ in self.models]
        cam = np.mean(cams,axis=0)
        resized = image.resize((224,224)).convert('RGB')
        pixels = np.asarray(resized).astype(float)
        # Spatial importance, not a segmentation mask or diagnostic localization.
        color = np.stack([cam, np.zeros_like(cam), 1-cam],axis=-1)*255
        overlay = Image.fromarray(np.uint8(np.clip(pixels*.65+color*.35,0,255)))
        buffer = io.BytesIO()
        overlay.save(buffer,format='PNG')
        confidence = float(probabilities[index])
        return {'task':task,'label':TASKS[task][index], 'model_score':confidence,
                'probabilities':{label:float(p) for label,p in zip(TASKS[task],probabilities)},
                'individual_models':{model.architecture:{label:float(p) for label,p in zip(TASKS[task],values)} for (model,_),values in zip(self.models,individual)},
                'review_suggested':confidence < .7 or int(individual[0].argmax()) != int(individual[1].argmax()),
                'review_note':'Heuristic review flag, not a validated safety threshold.',
                'gradcam_png':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                'explanation':'Mean Grad-CAM for the ensemble-predicted class; highlights influential regions, not lesion boundaries.',
                'latency_ms':round((time.perf_counter()-start)*1000),
                'training_status':[meta['status'] for _,meta in self.models],
                'cancer_risk':None,'clinical_fusion_used':False,
                'notice':'Uncalibrated research model scores are not a diagnosis or future cancer risk.'}


def decode_image(content):
    if not content:
        raise ValueError('Empty image')
    with Image.open(io.BytesIO(content)) as original:
        if original.format not in {'JPEG','PNG','BMP','TIFF'}:
            raise ValueError('Use JPEG, PNG, BMP, or TIFF')
        if original.width*original.height > 25_000_000:
            raise ValueError('Image exceeds 25 megapixels')
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.load()
    return image
