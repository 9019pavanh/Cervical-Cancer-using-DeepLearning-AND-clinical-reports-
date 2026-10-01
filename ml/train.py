"""Frozen ImageNet feature training, multi-task heads, untouched test evaluation."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from .catalog import TASKS
from .models import CytologyModel, preprocessing


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = preprocessing()
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        with Image.open(self.rows[index]['path']) as im:
            return self.transform(ImageOps.exif_transpose(im).convert('RGB')), index


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(output/'torch-cache'))
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if args.max_per_class:
        # Explicitly labeled pilot subset: sample each split independently without changing membership.
        random.shuffle(rows)
        counts, selected = {}, []
        for row in rows:
            key = (row['task'], row['label'], row['split'])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= args.max_per_class:
                selected.append(row)
        rows = selected
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for architecture in args.architectures:
        started = time.time()
        print(f'{architecture}: extracting features from {len(rows)} images on {device}', flush=True)
        model = CytologyModel(architecture, pretrained=True).to(device).eval()
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        cache = output/f'{architecture}-features.pt'
        signature = hashlib.sha256(''.join(r['sha256'] for r in rows).encode()).hexdigest()
        if cache.exists():
            stored = torch.load(cache, map_location='cpu', weights_only=True)
        else:
            stored = {}
        if stored.get('signature') == signature:
            features = stored['features']
        else:
            progress_cache = output/f'{architecture}-features.partial.pt'
            partial = torch.load(progress_cache,map_location='cpu',weights_only=True) if progress_cache.exists() else {}
            pieces = [partial['features']] if partial.get('signature') == signature else []
            completed = len(pieces[0]) if pieces else 0
            loader = DataLoader(Images(rows[completed:]), batch_size=args.batch_size, num_workers=args.workers)
            with torch.no_grad():
                for step, (images, _) in enumerate(loader):
                    pieces.append(model.backbone(images.to(device)).cpu())
                    if step % 10 == 0:
                        print(f'{architecture}: features {min(completed+(step+1)*args.batch_size,len(rows))}/{len(rows)}', flush=True)
                    if (step+1) % 50 == 0:
                        torch.save({'signature':signature,'features':torch.cat(pieces)},progress_cache)
            features = torch.cat(pieces)
            torch.save({'signature': signature, 'features': features}, cache)
        metrics, available = {}, []
        for task, classes in TASKS.items():
            ids = {split: [i for i,r in enumerate(rows) if r['task'] == task and r['split'] == split] for split in ['train','val','test']}
            if any(not ids[s] for s in ids):
                print(f'Skipping {task}: missing split', flush=True)
                continue
            x = {s: features[indices].to(device) for s,indices in ids.items()}
            y = {s: torch.tensor([classes.index(rows[i]['label']) for i in indices],device=device) for s,indices in ids.items()}
            if set(y['train'].tolist()) != set(range(len(classes))):
                print(f'Skipping {task}: missing training class', flush=True)
                continue
            head = model.heads[task]
            optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
            class_counts = torch.bincount(y['train'], minlength=len(classes)).float()
            loss_fn = torch.nn.CrossEntropyLoss(weight=(class_counts.sum()/class_counts.clamp_min(1)).to(device))
            best_loss, best_state, history = float('inf'), None, []
            for epoch in range(args.epochs):
                head.train()
                optimizer.zero_grad()
                loss = loss_fn(head(x['train']),y['train'])
                loss.backward()
                optimizer.step()
                head.eval()
                with torch.no_grad():
                    val_loss = loss_fn(head(x['val']),y['val']).item()
                history.append({'epoch':epoch+1,'train_loss':loss.item(),'val_loss':val_loss})
                if val_loss < best_loss:
                    best_loss, best_state = val_loss, copy.deepcopy(head.state_dict())
            head.load_state_dict(best_state)
            with torch.no_grad():
                pred = head(x['test']).argmax(1).cpu().numpy()
            truth = y['test'].cpu().numpy()
            metrics[task] = {'test_count':len(truth),'split_counts':{s:len(ids[s]) for s in ids},
                'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
                'classification_report':classification_report(truth,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),
                'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(len(classes)))).tolist(),'history':history}
            available.append(task)
            print(f'{architecture}/{task}: balanced accuracy {metrics[task]["balanced_accuracy"]:.3f}',flush=True)
        metadata = {'architecture':architecture,'tasks':available,'classes':TASKS,'manifest_sha256':manifest_hash,
            'status':'pilot' if args.max_per_class else 'research','max_per_class_per_split':args.max_per_class,
            'training':'ImageNet frozen backbone + trained task-specific linear heads', 'seed':args.seed,
            'elapsed_seconds':time.time()-started,'clinical_validation':False,'risk_model':False}
        checkpoint_path=output/f'{architecture}.pt'
        torch.save({'state_dict':model.cpu().state_dict(),'metadata':metadata},checkpoint_path)
        (output/f'{architecture}-metrics.json').write_text(json.dumps({'metadata':metadata,'metrics':metrics,
            'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
        print(f'Saved {architecture}: {time.time()-started:.1f}s',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',default='artifacts/manifest.json')
    parser.add_argument('--output',default='artifacts/models')
    parser.add_argument('--architectures',nargs='+',choices=['resnet50','efficientnet_b0'],default=['resnet50','efficientnet_b0'])
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=.003)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-per-class',type=int,default=0)
    args=parser.parse_args()
    if args.epochs < 1 or args.max_per_class < 0:
        parser.error('epochs must be positive and max-per-class nonnegative')
    train(args)
