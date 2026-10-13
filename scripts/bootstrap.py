"""Create isolated public reproduction environments; never alter shared packages."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'cee0c130a62a9c288ad8918f1f7cdf23cc228d69'

def run(argv):
    subprocess.run([str(v) for v in argv], check=True)

def environment(name, requirements, install):
    folder = ROOT / name
    if not folder.exists():
        run([sys.executable, '-m', 'venv', folder])
    python = folder / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if install:
        run([python, '-m', 'pip', 'install', '-r', ROOT / requirements])
    site = subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
    (Path(site) / 'semantic_research.pth').write_text(str(ROOT / 'src') + '\n' + str(ROOT / '.upstream/iris-3b/src') + '\n', encoding='utf-8')
    return python

def main(args):
    checkout = ROOT / '.upstream/iris-3b'
    if not checkout.exists():
        run(['git', 'clone', 'https://github.com/speridlabs/iris-3b.git', checkout])
    current = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if current != REVISION:
        if subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError('Upstream checkout has local changes; choose a fresh clone')
        run(['git', '-C', checkout, 'checkout', '--detach', REVISION])
    core = environment('.venv', 'requirements-full.txt', not args.no_install)
    if args.vlm:
        environment('.venv-vlm', 'requirements-vlm.txt', not args.no_install)
    print(json.dumps({'core_python': str(core), 'upstream_revision': REVISION, 'inference_run': False}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', action='store_true')
    parser.add_argument('--no-install', action='store_true')
    main(parser.parse_args())
