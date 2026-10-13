"""Resume a pinned public checkpoint with checked HTTP byte ranges and final SHA-256."""
import concurrent.futures
import hashlib
import json
import math
import os
import threading
import time
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"sources/model-manifest.json").read_text())
revision=manifest["revision"]
filename="model.safetensors"
url=f"https://huggingface.co/speridlabs/iris-3b/resolve/{revision}/{filename}"
req=urllib.request.Request(f"https://huggingface.co/api/models/speridlabs/iris-3b/revision/{revision}?blobs=true",
                           headers={"User-Agent":"SemanticCommunicationResearch/0.1"})
with urllib.request.urlopen(req,timeout=40) as response:
    info=json.load(response)
entry=next(item for item in info["siblings"] if item["rfilename"]==filename)
total=entry["size"]
expected_sha=entry["lfs"]["sha256"]
target=ROOT/"models/iris-3b/model.safetensors"
parts=target.parent/"model-parts"
parts.mkdir(exist_ok=True)
chunk=32*1024*1024
count=math.ceil(total/chunk)
lock=threading.Lock()
progress={"revision":revision,"expected_bytes":total,"expected_sha256":expected_sha,
          "chunk_bytes":chunk,"chunks":count,"workers":4,"completed_chunks":0,"downloaded_bytes":0,"started_at":time.time()}
def save_progress():
    path=ROOT/"results/download-progress.json"
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(progress,indent=2),encoding="utf-8")
    temp.replace(path)
def download(index):
    start=index*chunk
    stop=min(total-1,start+chunk-1)
    length=stop-start+1
    path=parts/f"{index:06d}.part"
    if not path.is_file() or path.stat().st_size!=length:
        error=None
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={"Range":f"bytes={start}-{stop}",
                    "User-Agent":"SemanticCommunicationResearch/0.1","Accept-Encoding":"identity"})
                with urllib.request.urlopen(request,timeout=45) as response:
                    if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{stop}/{total}":
                        raise RuntimeError("Server did not honor the exact byte range")
                    with open(path,"wb") as output:
                        remaining=length
                        while remaining:
                            block=response.read(min(1024*1024,remaining))
                            if not block:
                                raise RuntimeError("Truncated byte range")
                            output.write(block)
                            remaining-=len(block)
                if path.stat().st_size!=length:
                    raise RuntimeError("Incorrect part size")
                break
            except Exception as exc:
                error=exc
                print(json.dumps({"chunk":index,"attempt":attempt+1,"error":str(exc)}),flush=True)
                time.sleep(min(5,attempt+1))
        else:
            raise RuntimeError(f"Chunk {index} failed after retries: {error}")
    with lock:
        progress["completed_chunks"]+=1
        progress["downloaded_bytes"]+=length
        progress["elapsed_seconds"]=time.time()-progress["started_at"]
        save_progress()
        if index%10==0 or progress["completed_chunks"]==count:
            print(json.dumps({"parts":progress["completed_chunks"],"total_parts":count,
                 "GiB":progress["downloaded_bytes"]/2**30,"seconds":progress["elapsed_seconds"]}),flush=True)
    return path
print(json.dumps(progress),flush=True)
save_progress()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    paths=list(pool.map(download,range(count)))
print("Assembling and checking SHA-256",flush=True)
digest=hashlib.sha256()
assembly=target.with_suffix(".assembled")
with open(assembly,"wb") as output:
    for path in paths:
        with open(path,"rb") as source:
            for block in iter(lambda:source.read(8*1024*1024),b""):
                output.write(block)
                digest.update(block)
actual=digest.hexdigest()
if assembly.stat().st_size!=total or actual!=expected_sha:
    raise RuntimeError("Checkpoint integrity verification failed")
assembly.replace(target)
progress.update({"phase":"complete","sha256":actual,"total_seconds":time.time()-progress["started_at"]})
save_progress()
print(json.dumps(progress),flush=True)
# Keep resumable parts until explicit archival/cleanup; no destructive file deletion here.
