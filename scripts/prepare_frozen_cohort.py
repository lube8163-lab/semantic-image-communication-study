"""Prepare the frozen published cohort without any other project's directories."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from text_route_common import ROOT, RUN, PROTOCOL, MODEL_PATHS, read, save, sha, now

def main():
    dataset = read(ROOT / 'sources/published-dataset.json')
    for case in dataset['images']:
        if sha(ROOT / case['file'].replace('\\', '/')) != case['sha256']:
            raise ValueError('Frozen published input changed: ' + case['id'])
    dataset.update({'created_utc': now(), 'protocol_sha256': sha(PROTOCOL), 'reproduction': 'Frozen published cohort; no fresh population/benchmark sample'})
    RUN.mkdir(parents=True, exist_ok=True)
    previous = RUN / 'dataset.json'
    if previous.exists():
        old = read(previous)
        if old['protocol_sha256'] != dataset['protocol_sha256'] or [(r['id'], r['sha256']) for r in old['images']] != [(r['id'], r['sha256']) for r in dataset['images']]:
            raise ValueError('Existing run identity differs; choose a new SC_RUN')
    else:
        save(previous, dataset)
    assets = read(ROOT / 'sources/text-route-cached-assets.json')
    for name in ['minicpm', 'lfm16', 'dino']:
        folder = MODEL_PATHS[name] if name in MODEL_PATHS else ROOT / 'models/dino'
        for entry in assets[name]['files']:
            path = folder / entry['path']
            if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError('Pinned asset changed: ' + name + '/' + entry['path'])
    print('FROZEN_INPUTS_VERIFIED', len(dataset['images']), flush=True)

if __name__ == '__main__':
    main()
