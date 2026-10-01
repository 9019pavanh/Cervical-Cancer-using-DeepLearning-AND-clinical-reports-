"""Evaluate equal-weight ensemble using cached, held-out image features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from .catalog import TASKS


def evaluate(manifest, directory):
    directory=Path(directory)
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'))
    manifest_hash=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    signature=hashlib.sha256(''.join(r['sha256'] for r in rows).encode()).hexdigest()
    outputs=[]
    for architecture in ['resnet50','efficientnet_b0']:
        checkpoint=torch.load(directory/f'{architecture}.pt',map_location='cpu',weights_only=True)
        metadata=checkpoint['metadata']
        if metadata.get('manifest_sha256') != manifest_hash:
            raise ValueError(f'{architecture}: checkpoint was trained on a different manifest or split.')
        if metadata.get('status') != 'research':
            raise ValueError('Full-cohort ensemble evaluation requires research checkpoints, not pilot checkpoints.')
        if metadata.get('architecture') != architecture or metadata.get('classes') != TASKS:
            raise ValueError(f'{architecture}: incompatible checkpoint architecture or class definitions.')
        cache=torch.load(directory/f'{architecture}-features.pt',map_location='cpu',weights_only=True)
        if cache['signature'] != signature:
            raise ValueError('Feature cache must match the full manifest. Pilot caches cannot be used here.')
        outputs.append((checkpoint,cache['features']))
    metrics={}
    for task,classes in TASKS.items():
        indices=[i for i,r in enumerate(rows) if r['task']==task and r['split']=='test']
        if not indices or any(task not in saved['metadata']['tasks'] for saved,_ in outputs):
            continue
        scores=[]
        for saved,features in outputs:
            state=saved['state_dict']
            logits=torch.nn.functional.linear(features[indices],state[f'heads.{task}.weight'],state[f'heads.{task}.bias'])
            scores.append(logits.softmax(1).numpy())
        scores=np.mean(scores,axis=0)
        truth=np.array([classes.index(rows[i]['label']) for i in indices])
        prediction=scores.argmax(1)
        metrics[task]={'test_count':len(indices),'balanced_accuracy':float(balanced_accuracy_score(truth,prediction)),
            'classification_report':classification_report(truth,prediction,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(truth,prediction,labels=list(range(len(classes)))).tolist()}
    result={'metadata':{'architecture':'equal_weight_ensemble','status':'research','clinical_validation':False,
        'manifest_sha256':manifest_hash,
        'checkpoint_sha256':{architecture:hashlib.sha256((directory/f'{architecture}.pt').read_bytes()).hexdigest() for architecture in ['resnet50','efficientnet_b0']}},'metrics':metrics}
    (directory/'ensemble-metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({task:{'balanced_accuracy':m['balanced_accuracy'],'test_count':m['test_count']} for task,m in metrics.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='artifacts/manifest.json')
    parser.add_argument('--directory',default='artifacts/models')
    args=parser.parse_args()
    evaluate(args.manifest,args.directory)
