"""Pinned, resumable public HF downloads without shared caches or credentials."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "SemanticCommunicationResearch/0.2", "Accept-Encoding": "identity"}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def run(args):
    folder = ROOT / "models" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        metadata = json.load(response)
    if metadata["sha"] != args.revision:
        raise ValueError("Unexpected repository revision")
    selected = [v for v in metadata["siblings"] if "/" not in v["rfilename"] and
                (v["rfilename"].endswith((".json", ".txt", ".md", ".jinja", ".safetensors")) or v["rfilename"] == "LICENSE")]
    manifest = {"repo": args.repo, "revision": args.revision, "license": metadata.get("cardData", {}).get("license"), "files": []}
    progress = {"repo": args.repo, "revision": args.revision, "started_unix": time.time(), "phase": "download", "files": {}}
    lock = threading.Lock()
    progress_path = ROOT / "results" / f"download-{args.name}.json"
    def save():
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        temporary.replace(progress_path)
    for entry in selected:
        name = entry["rfilename"]
        target = folder / name
        total = entry["size"]
        expected = entry.get("lfs", {}).get("sha256")
        url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}"
        print(json.dumps({"file": name, "bytes": total}), flush=True)
        progress["files"][name] = {"expected_bytes": total, "downloaded_bytes": 0, "phase": "download"}
        save()
        if target.exists() and target.stat().st_size == total and (not expected or sha(target) == expected):
            pass
        elif total < 16 * 1024 * 1024:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                data = response.read()
            if len(data) != total:
                raise ValueError(f"Wrong file size: {name}")
            target.write_bytes(data)
        else:
            chunk = 32 * 1024 * 1024
            count = math.ceil(total / chunk)
            parts = folder / (name + ".parts")
            parts.mkdir(exist_ok=True)
            def download(index):
                start = index * chunk
                stop = min(total - 1, start + chunk - 1)
                length = stop - start + 1
                part = parts / f"{index:06d}.part"
                if not part.is_file() or part.stat().st_size != length:
                    for attempt in range(5):
                        try:
                            headers = dict(HEADERS, Range=f"bytes={start}-{stop}")
                            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=50) as response:
                                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{total}":
                                    raise ValueError("Server did not honor exact byte range")
                                temporary = part.with_suffix(".tmp")
                                with open(temporary, "wb") as output:
                                    remaining = length
                                    while remaining:
                                        block = response.read(min(1024 * 1024, remaining))
                                        if not block:
                                            raise ValueError("Truncated byte range")
                                        output.write(block)
                                        remaining -= len(block)
                                temporary.replace(part)
                            break
                        except Exception as exc:
                            print(json.dumps({"file": name, "part": index, "attempt": attempt + 1, "error": str(exc)}), flush=True)
                            if attempt == 4:
                                raise
                            time.sleep(min(5, attempt + 1))
                with lock:
                    progress["files"][name]["downloaded_bytes"] += length
                    progress["elapsed_seconds"] = time.time() - progress["started_unix"]
                    save()
                    if index % 10 == 0 or index + 1 == count:
                        print(json.dumps({"file": name, **progress["files"][name], "elapsed_seconds": progress["elapsed_seconds"]}), flush=True)
                return part
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(download, range(count)))
            assembled = target.with_suffix(".assembled")
            with open(assembled, "wb") as output:
                for part in paths:
                    with open(part, "rb") as source:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            output.write(block)
            if assembled.stat().st_size != total or (expected and sha(assembled) != expected):
                raise ValueError(f"Integrity verification failed: {name}")
            assembled.replace(target)
        digest = sha(target)
        if expected and digest != expected:
            raise ValueError(f"Wrong SHA-256: {name}")
        manifest["files"].append({"name": name, "bytes": total, "sha256": digest, "lfs_sha256": expected})
        progress["files"][name].update({"downloaded_bytes": total, "sha256": digest, "phase": "complete"})
        save()
    (ROOT / "sources" / f"{args.name}-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress.update({"phase": "complete", "elapsed_seconds": time.time() - progress["started_unix"]})
    save()
    print(json.dumps(progress), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--name", required=True)
    run(p.parse_args())
