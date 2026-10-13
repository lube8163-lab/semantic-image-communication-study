"""Download exact separately licensed files and verify published SHA-256 values."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def assets():
    iris = json.loads((ROOT / 'sources/model-manifest.json').read_text('utf-8'))
    qwen = json.loads((ROOT / 'sources/qwen3-vl-4b-manifest.json').read_text('utf-8'))
    cached = json.loads((ROOT / 'sources/text-route-cached-assets.json').read_text('utf-8'))
    iris = {**iris, 'directory': 'iris-3b', 'files': [{**f, 'path': f['file'].replace('\\', '/').split('models/iris-3b/', 1)[1]} for f in iris['files']]}
    qwen = {**qwen, 'directory': 'qwen3-vl-4b', 'files': [{**f, 'path': f['name']} for f in qwen['files']]}
    return {'iris': iris, 'qwen4b': qwen, **{n: {**v, 'directory': n} for n, v in cached.items()}, 'clip': json.loads((ROOT / 'sources/clip-evaluator-assets.json').read_text('utf-8'))}

def main(args):
    manifest = assets()[args.model]
    if args.list:
        print(json.dumps(manifest, ensure_ascii=False, indent=2)); return
    folder = ROOT / 'models' / manifest['directory']
    for entry in manifest['files']:
        relative = entry['path']
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) == entry['sha256']:
            print('VERIFIED', relative, flush=True); continue
        if args.verify_only:
            raise ValueError('Missing or changed asset: ' + relative)
        url = entry.get('url') or f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{relative}"
        request = urllib.request.Request(url, headers={'User-Agent': 'SemanticImageCommunicationStudy/1.0'})
        temporary = target.with_suffix(target.suffix + '.download')
        print('DOWNLOADING', relative, entry['bytes'], flush=True)
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                output.write(block)
        if temporary.stat().st_size != entry['bytes'] or sha(temporary) != entry['sha256']:
            raise ValueError('Downloaded asset hash differs; original target preserved')
        temporary.replace(target)
        print('VERIFIED', relative, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['iris', 'qwen4b', 'minicpm', 'lfm16', 'dino', 'clip'], required=True)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    main(parser.parse_args())
