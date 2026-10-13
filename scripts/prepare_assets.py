"""Download exactly one trained checkpoint, one null prompt, and 3 public images."""
import hashlib
import json
import os
import sys
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "models" / ".hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageOps

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "SemanticCommunicationResearch/0.1"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()

def main():
    model_dir = ROOT / "models" / "iris-3b"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch("https://huggingface.co/api/models/speridlabs/iris-3b?blobs=true"))
    prior_manifest = ROOT / "sources/model-manifest.json"
    revision = json.loads(prior_manifest.read_text())["revision"] if prior_manifest.is_file() else metadata["sha"]
    size_map = {x["rfilename"]: x.get("size", 0) for x in metadata["siblings"]}
    manifest = {"repo": "speridlabs/iris-3b", "revision": revision, "license": "Apache-2.0",
                "source_revision": subprocess.check_output(["git", "-C", str(ROOT / ".upstream/iris-3b"), "rev-parse", "HEAD"], text=True).strip(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "files": []}
    selected = ["config.yaml", "upscaler/empty_prompt.safetensors", "model.safetensors"]
    print(json.dumps({"revision": revision, "selected_files": selected,
                      "declared_download_bytes": sum(size_map.get(f, 0) for f in selected)}), flush=True)
    for filename in ([] if "--images-only" in sys.argv else selected):
        print("Downloading " + filename, flush=True)
        path = Path(hf_hub_download(repo_id=manifest["repo"], filename=filename, revision=revision,
                                   local_dir=model_dir, token=False))
        entry = {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest["files"].append(entry)
        (ROOT / "sources/model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(entry), flush=True)
    original_dir, processed_dir = ROOT / "data/originals", ROOT / "data/processed"
    original_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ("astronaut", "NASA / Eileen Collins", "Public domain"),
        ("coffee", "Rachel Michetti / Pikolo Espresso Bar", "CC0"),
        ("chelsea", "Stefan van der Walt", "CC0"),
    ]
    images = []
    for name, credit, license_name in samples:
        url = f"https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/{name}.png"
        path = original_dir / f"{name}.png"
        if not path.is_file():
            path.write_bytes(fetch(url))
        image = ImageOps.fit(Image.open(path).convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS)
        output = processed_dir / f"{name}.png"
        image.save(output)
        images.append({"id": name, "kind": "photograph", "source": url, "credit": credit,
                       "license": license_name, "license_source": "https://scikit-image.org/docs/stable/api/skimage.data.html#" + name,
                       "original_sha256": sha256(path), "original_bytes": path.stat().st_size,
                       "file": str(output.relative_to(ROOT)), "sha256": sha256(output),
                       "processing": "RGB, center crop to square and Lanczos resize to 256x256"})
    for name, color, shape, side in [("red_circle_left", (215,45,45), "circle", "left"),
                                     ("blue_square_right", (45,85,215), "square", "right")]:
        image = Image.new("RGB", (256,256), (240,235,225))
        draw = ImageDraw.Draw(image)
        box = (28,72,116,160) if side == "left" else (140,72,228,160)
        (draw.ellipse if shape == "circle" else draw.rectangle)(box, fill=color)
        draw.line((0,208,255,208), fill=(90,90,90), width=4)
        output = processed_dir / f"{name}.png"
        image.save(output)
        images.append({"id": name, "kind": "synthetic_control", "source": "scripts/prepare_assets.py",
                       "license": "CC0", "attributes": {"color":color,"shape":shape,"side":side},
                       "file": str(output.relative_to(ROOT)), "sha256":sha256(output)})
    (ROOT / "data/manifest.json").write_text(json.dumps({"images":images, "resolution":[256,256]}, indent=2), encoding="utf-8")
    print("Prepared 3 photographs and 2 synthetic controls.", flush=True)

if __name__ == "__main__":
    main()
