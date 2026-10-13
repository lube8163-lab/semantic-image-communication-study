"""Verify public artifacts and measured numbers without a GPU or third-party package."""
import ast
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/text-route-20261013'

def read(path):
    return json.loads(path.read_text('utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    failures = []
    def require(condition, message):
        if not condition:
            failures.append(message)
    plan = read(ROOT / 'configs/text-route.json')
    dataset = read(RUN / 'dataset.json')
    require(len(dataset['images']) == 13, 'Input coverage')
    for case in dataset['images']:
        path = ROOT / case['file'].replace('\\', '/')
        require(path.exists() and sha(path) == case['sha256'], 'Input hash: ' + case['id'])
    require(sha(ROOT / 'configs/text-route.json') == dataset['protocol_sha256'], 'Frozen protocol hash')
    captions = [v for model in plan['models'] for v in read(RUN / 'captions' / (model + '.json'))]
    require(len(captions) == 78 and all(v['status'] == 'completed' for v in captions), 'Caption coverage')
    require(all(v['prompt'] == v['raw_output'] for v in captions), 'Raw captions were edited')
    records = read(RUN / 'generation.json') + read(RUN / 'generation-refine40.json')
    require(len(records) == 176, 'Generation coverage')
    for row in records:
        for path_key, hash_key in [('file', 'sha256'), ('native_file', 'native_sha256'), ('packet_file', 'packet_sha256')]:
            path = RUN / row[path_key]
            require(path.exists() and sha(path) == row[hash_key], 'Output hash: ' + row[path_key])
        packet = RUN / row['packet_file']
        require(packet.stat().st_size == row['wire_bytes'], 'Wire byte count')
        content = read(packet)
        require(set(content) == {'protocol', 'prompt', 'seed', 'width', 'height'} and content['prompt'] == row['prompt'], 'Independent text packet schema')
    metrics = read(RUN / 'metrics.json')
    require(len(metrics) == 459, 'Metric coverage')
    for row in metrics:
        require((ROOT / row['path']).exists(), 'Metric image missing: ' + row['path'])
    summary = read(RUN / 'evaluation-summary.json')
    keys = ['wire_bytes', 'clip_image_cosine', 'dino_image_cosine', 'psnr_db', 'mae_rgb8']
    for model in plan['models']:
        for budget in plan['budgets']:
            group = [r for r in metrics if r['route'] == 'text_only' and r['kind'] == 'photograph' and r['model'] == model and r['budget'] == budget]
            require(len(group) == 22 and len({r['case_id'] for r in group}) == 11, 'Photo/seed grid')
            aggregate = next(r for r in summary['photo_aggregates'] if (r['route'], r['model'], r['budget']) == ('text_only', model, budget))
            for key in keys:
                per_image = [statistics.mean(r[key] for r in group if r['case_id'] == case_id) for case_id in sorted({r['case_id'] for r in group})]
                require(abs(statistics.mean(per_image) - aggregate[key]) < 1e-10, 'Recorded mean changed: ' + model + '/' + budget + '/' + key)
    html_references = 0
    for report in [RUN / 'report.html', ROOT / 'results/pilot-20261013/report.html']:
        for target in re.findall(r'(?:href|src)="([^"#]+)"', report.read_text('utf-8')):
            if target.startswith(('http://', 'https://', 'data:')):
                continue
            path = report.parent / target
            require(path.exists(), 'Report reference missing: ' + str(path.relative_to(ROOT)))
            html_references += 1
    forbidden_extensions = {'.safetensors', '.pt', '.pth', '.ckpt', '.gguf', '.onnx', '.part', '.bin'}
    private = re.compile(r'C:[/\\]+Users[/\\]|D:[/\\]+AI[/\\]|GPU-[0-9a-f-]{36}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----', re.I)
    count = 0
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p.startswith('.') or p in ['models', 'site', 'publication-local', '__pycache__'] for p in rel.parts):
            continue
        count += 1
        require(path.suffix.lower() not in forbidden_extensions, 'Model/binary extension: ' + str(rel))
        require(path.stat().st_size < 50 * 1024 * 1024, 'Oversized public file: ' + str(rel))
        if path.suffix.lower() in ['.py', '.ps1', '.json', '.csv', '.md', '.html', '.txt'] and path.name != 'verify_publication.py':
            require(not private.search(path.read_text('utf-8-sig')), 'Private path/identifier/credential pattern: ' + str(rel))
        if path.suffix == '.py':
            ast.parse(path.read_text('utf-8-sig'), filename=str(rel))
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps({'status': 'passed', 'inputs': 13, 'raw_captions': 78, 'generation_and_packet_hash_pairs': 176, 'metric_rows': 459, 'photo_aggregate_values_recomputed': 30, 'local_report_references': html_references, 'public_files_scanned': count, 'fresh_inference': False}, indent=2))

if __name__ == '__main__':
    main()
