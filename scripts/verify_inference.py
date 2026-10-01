"""Run real checkpoint/API checks on one held-out image per task."""
import base64
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app


def verify():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'artifacts'/'manifest.json').read_text(encoding='utf-8'))
    output=root/'artifacts'/'smoke'
    output.mkdir(exist_ok=True)
    results=[]
    with TestClient(app) as client:
        status=client.get('/api/health').json()
        assert status['ready'], 'Both trained checkpoints are required'
        assert len(status['models'])==2
        assert not status['clinical_fusion_available']
        for task in status['tasks']:
            sample=next(row for row in manifest if row['task']==task and row['split']=='test')
            with open(sample['path'],'rb') as handle:
                response=client.post('/api/predict',data={'task':task},files={'file':('cytology-image',handle)})
            assert response.status_code==200, response.text
            result=response.json()
            assert abs(sum(result['probabilities'].values())-1)<1e-5
            assert len(result['individual_models'])==2
            assert result['cancer_risk'] is None and result['clinical_fusion_used'] is False
            overlay=base64.b64decode(result['gradcam_png'].split(',',1)[1])
            with Image.open(io.BytesIO(overlay)) as im:
                assert im.size==(224,224)
                im.verify()
            (output/f'{task}-gradcam.png').write_bytes(overlay)
            results.append({'task':task,'reference_label':sample['label'],'predicted_label':result['label'],
                            'model_score':result['model_score'],'latency_ms':result['latency_ms'],
                            'status_code':response.status_code})
        assert client.post('/api/predict',files={'file':('invalid.png',b'bad')}).status_code==400
        metrics=client.get('/api/metrics').json()
        assert len(metrics)>=3, 'Both individual and ensemble reports are required'
    (output/'inference-smoke.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    verify()
