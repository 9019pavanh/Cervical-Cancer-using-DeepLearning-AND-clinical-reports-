"""Audit images and retain distinct, incompatible cytology label taxonomies."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

TASKS = {
    'bethesda': ['ASCUS', 'HSIL', 'LSIL', 'NILM', 'SCC'],
    'sipakmed': ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate'],
    'herlev': ['carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic', 'normal_columnar', 'normal_intermediate', 'normal_superficiel', 'severe_dysplastic'],
    'binary': ['Abnormal', 'Normal'],
}
ALIASES = {'High squamous intra-epithelial lesion': 'HSIL', 'Low squamous intra-epithelial lesion': 'LSIL',
           'Negative for Intraepithelial malignancy': 'NILM', 'Squamous cell carcinoma': 'SCC', 'NL': 'NILM',
           'Abnormal_Renamed': 'Abnormal', 'Normal_Renamed': 'Normal'}


def identify(path):
    for part in reversed(path.parts[:-1]):
        label = ALIASES.get(part, part.removeprefix('im_'))
        for task, classes in TASKS.items():
            if label in classes:
                return task, label
    return None


def image_group(path, source, task, label):
    # Filename-derived slide proxies; these are NOT verified patient identities.
    stem = path.stem
    if task == 'sipakmed':
        return f'sipakmed:{label}:{stem.split("_")[0]}'
    if task == 'herlev':
        return f'herlev:{stem.rsplit("-", 1)[0]}'
    if source == 'Cytolog Cervical Cancer':
        match = re.match(r'([A-Za-z]+\d{3})', stem)
        if match:
            return f'cytolog:{match[1]}'
    return f'{source}:{task}:{label}:{stem}'


def build(config, output, seed=42):
    roots = json.loads(Path(config).read_text(encoding='utf-8-sig'))
    records, skipped = [], []
    for source, root in roots.items():
        print(f'Auditing {source}...', flush=True)
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset directory missing: {root}')
        for path in sorted(root.rglob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                continue
            found = identify(path)
            if not found:
                skipped.append({'path': str(path), 'reason': 'unrecognized label'})
                continue
            task, label = found
            # SIPaKMeD fields and isolated crops are different units: use isolated crops only.
            if task == 'sipakmed' and 'cropped' not in [p.lower() for p in path.parts]:
                skipped.append({'path': str(path), 'reason': 'whole-field SIPaKMeD image'})
                continue
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            except Exception as exc:
                skipped.append({'path': str(path), 'reason': str(exc)})
                continue
            records.append({'path': str(path), 'source': source, 'task': task, 'label': label,
                            'sha256': digest, 'group': image_group(path, source, task, label)})
            if len(records) % 1000 == 0:
                print(f'Checked {len(records)} labeled images', flush=True)
    # Union duplicate images AND their slide proxies BEFORE splitting.
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        parents[find(a)] = find(b)
    duplicates = defaultdict(list)
    for row in records:
        find(row['group'])
        duplicates[row['sha256']].append(row)
    conflicts = []
    for digest, rows in duplicates.items():
        for row in rows[1:]:
            union(rows[0]['group'], row['group'])
        if len({(r['task'], r['label']) for r in rows}) > 1:
            conflicts.append(digest)
    rejected = set(conflicts)
    unique = [rows[0] for digest, rows in duplicates.items() if digest not in rejected]
    groups = sorted({find(r['group']) for r in unique})
    random.Random(seed).shuffle(groups)
    splits = {g: ('train' if i < len(groups)*.7 else 'val' if i < len(groups)*.85 else 'test') for i, g in enumerate(groups)}
    for row in unique:
        row['group'] = find(row['group'])
        row['split'] = splits[row['group']]
    report = {'scanned_labeled_images': len(records), 'unique_images': len(unique),
              'duplicates_removed': sum(len(v)-1 for v in duplicates.values()), 'conflicting_hashes_excluded': len(conflicts),
              'skipped': len(skipped), 'seed': seed,
              'counts': dict(Counter(f"{r['task']}/{r['label']}/{r['split']}" for r in unique)),
              'sources': dict(Counter(r['source'] for r in records)),
              'limitations': ['Slide groups inferred from filenames; patient-level leakage remains unverified.',
                              'Pixel-exact duplicates removed; transformed duplicates may remain.',
                              'No clinical variables or longitudinal cancer-risk outcomes supplied.']}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(unique, indent=2), encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output/'excluded.json').write_text(json.dumps(skipped, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/datasets.local.json')
    parser.add_argument('--output', default='artifacts')
    args = parser.parse_args()
    build(args.config, args.output)
