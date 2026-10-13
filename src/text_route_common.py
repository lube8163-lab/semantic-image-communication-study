"""Non-GPU identity, dataset and wire-packet helpers for E004."""
import hashlib
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / os.environ.get("SC_RUN", "results/text-route-20261013")
PROTOCOL = ROOT / os.environ.get("SC_PROTOCOL", "configs/text-route.json")
MODEL_PATHS = {
    "qwen4b": Path(os.environ.get("SC_MODEL_QWEN4B", str(ROOT / "models/qwen3-vl-4b"))),
    "minicpm": Path(os.environ.get("SC_MODEL_MINICPM", str(ROOT / "models/minicpm"))),
    "lfm16": Path(os.environ.get("SC_MODEL_LFM16", str(ROOT / "models/lfm16"))),
}

def read(path):
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def protocol():
    return read(PROTOCOL)

def cases():
    return read(RUN / "dataset.json")["images"]

def wire(prompt, seed):
    p = protocol()
    value = {"protocol": p["experiment"], "prompt": prompt, "seed": seed, "width": p["width"], "height": p["height"]}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def encoding_key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def caption_path(name):
    return RUN / "captions" / f"{name}.json"

def all_captions():
    return [r for name in protocol()["models"] for r in read(caption_path(name))]
