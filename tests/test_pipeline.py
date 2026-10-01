import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.service import decode_image
from ml.catalog import TASKS, build, identify
from ml.models import CytologyModel, gradcam, preprocessing


def test_taxonomies_remain_separate(tmp_path):
    assert identify(tmp_path/'im_Metaplastic'/'1.bmp') == ('sipakmed','Metaplastic')
    assert identify(tmp_path/'High squamous intra-epithelial lesion'/'1.jpg') == ('bethesda','HSIL')
    assert identify(tmp_path/'carcinoma_in_situ'/'1.bmp') == ('herlev','carcinoma_in_situ')


def test_duplicate_groups_cannot_cross_splits(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    for root in [a,b]:
        (root/'HSIL').mkdir(parents=True)
    for i in range(20):
        image=Image.new('RGB',(8,8),(i,20,30))
        image.save(a/'HSIL'/f'{i}.png')
        image.save(b/'HSIL'/f'copy{i}.png')
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'a':str(a),'b':str(b)}))
    build(config,tmp_path/'out')
    rows=json.loads((tmp_path/'out'/'manifest.json').read_text())
    assert len(rows)==20
    assert len({r['sha256'] for r in rows})==20
    groups={}
    for row in rows:
        groups.setdefault(row['group'],set()).add(row['split'])
    assert all(len(values)==1 for values in groups.values())


def test_conflicting_labels_are_excluded(tmp_path):
    for label in ['HSIL','LSIL']:
        (tmp_path/label).mkdir()
        Image.new('RGB',(8,8)).save(tmp_path/label/'same.png')
    config=tmp_path/'config.json';config.write_text(json.dumps({'source':str(tmp_path)}))
    build(config,tmp_path/'out')
    assert json.loads((tmp_path/'out'/'manifest.json').read_text())==[]


def test_duplicates_join_entire_slide_groups(tmp_path):
    a,b=tmp_path/'cytolog',tmp_path/'other'
    for root in [a,b]:
        (root/'HSIL').mkdir(parents=True)
    Image.new('RGB',(8,8),(10,20,30)).save(a/'HSIL'/'A00101.png')
    Image.new('RGB',(8,8),(20,30,40)).save(a/'HSIL'/'A00102.png')
    Image.new('RGB',(8,8),(10,20,30)).save(b/'HSIL'/'same.png')
    config=tmp_path/'config.json';config.write_text(json.dumps({'Cytolog Cervical Cancer':str(a),'other':str(b)}))
    build(config,tmp_path/'out')
    rows=json.loads((tmp_path/'out'/'manifest.json').read_text())
    assert len(rows)==2
    assert len({r['group'] for r in rows})==1
    assert len({r['split'] for r in rows})==1


def test_api_rejects_invalid_oversized_and_unsupported_inputs(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_DIR',str(tmp_path))
    class ReadyPredictor:
        def status(self):
            return {'ready':True}
        def predict(self,image,task):
            raise ValueError('Unsupported task')
    with TestClient(app) as client:
        app.state.predictor=ReadyPredictor()
        assert client.post('/api/predict',files={'file':('bad.png',b'invalid')}).status_code==400
        assert client.post('/api/predict',files={'file':('large.png',b'x'*(10*1024*1024+1))}).status_code==413
        buffer=io.BytesIO();Image.new('RGB',(8,8)).save(buffer,format='PNG')
        assert client.post('/api/predict',files={'file':('valid.png',buffer.getvalue())},data={'task':'unknown'}).status_code==422


def test_invalid_upload_and_missing_models(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_DIR',str(tmp_path))
    with TestClient(app) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/health').json()['ready'] is False
        assert client.post('/api/predict',files={'file':('bad.png',b'invalid')}).status_code==503
    with pytest.raises(Exception):
        decode_image(b'invalid')


def test_architectures_and_explanations():
    torch.set_num_threads(2)
    tensor=preprocessing()(Image.new('RGB',(224,224),(150,80,100)))[None]
    for architecture in ['resnet50','efficientnet_b0']:
        model=CytologyModel(architecture).eval()
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            assert model(tensor,'bethesda').shape==(1,len(TASKS['bethesda']))
        cam=gradcam(model,tensor,'bethesda',0)
        assert cam.shape==(224,224)
        assert np.isfinite(cam).all()
        assert cam.min()>=0 and cam.max()<=1
