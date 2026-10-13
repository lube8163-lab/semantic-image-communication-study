"""Independent packet receiver using full frozen Iris rectified-flow sampling."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SC_GENERATOR_GPU", "0"))
os.environ.setdefault("IRIS_ATTENTION_BACKEND", "torch_cudnn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import load_file
from iris3b.sampling import generate
from iris3b.text.base import TextEncoding
from PIL import Image
from research_runtime import public_noise, tensor_image
from text_generator_runtime import load_iris, guard
from text_route_common import ROOT, RUN, PROTOCOL, protocol, read, save, sha, now, encoding_key

class PacketTextEncoder:
    def __init__(self, records):
        self.expected = {r["key"]: r["sha256"] for r in records}
        self.checked = set()

    def encode(self, prompts):
        if len(prompts) != 1:
            raise ValueError("One packet per reconstruction")
        path = RUN / "text-embeddings" / (encoding_key(prompts[0]) + ".safetensors")
        if path.name not in self.checked:
            if sha(path) != self.expected[encoding_key(prompts[0])]:
                raise ValueError("Receiver-local text cache changed")
            self.checked.add(path.name)
        tensors = load_file(path)
        return TextEncoding(**tensors)

    def null(self, negative_prompt=""):
        if negative_prompt:
            raise ValueError("Unexpected negative prompt")
        return self.encode([""])

def main(args):
    plan = protocol()
    encodings = read(RUN / "text-encoding.json")
    if any(r["status"] != "completed" or r["protocol_sha256"] != sha(PROTOCOL) for r in encodings):
        raise ValueError("Incomplete or changed decoder protocol")
    items = [(row, packet) for row in encodings for packet in row["packets"]]
    steps = plan["steps"]
    protocol_id = plan["experiment"]
    if args.refine:
        check = plan["refinement_check"]
        steps = check["steps"]
        protocol_id = check["protocol"]
        items = [v for v in items if v[0]["case_id"] in check["case_ids"] and v[1]["seed"] == check["seed"]]
    if args.smoke:
        items = [v for v in items if v[0]["model"] != "empty_control"][:2]
    suffix = "-smoke" if args.smoke else "-refine40" if args.refine else ""
    destination = RUN / ("generation" + suffix + ".json")
    records = read(destination) if destination.exists() else []
    done = {r["packet_file"] for r in records if r["status"] == "completed"}
    start = time.perf_counter()
    model, model_config, model_manifest = load_iris()
    print("IRIS_READY", time.perf_counter() - start, flush=True)
    encoder = PacketTextEncoder(encodings)
    for i, (row, packet) in enumerate(items):
        if packet["file"] in done:
            continue
        path = RUN / packet["file"]
        if sha(path) != packet["sha256"] or path.stat().st_size != packet["bytes"]:
            raise ValueError("Wire packet changed")
        value = json.loads(path.read_bytes())
        if args.refine:
            value["protocol"] = protocol_id
            path = RUN / "text-packets-refine40" / path.name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            packet = {"file": path.relative_to(RUN).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path), "seed": value["seed"]}
        if value["protocol"] != protocol_id or value["width"] != plan["width"] or value["height"] != plan["height"]:
            raise ValueError("Unsupported packet protocol")
        record = {"at": now(), "model": row["model"], "case_id": row["case_id"], "budget": row["budget"], "seed": value["seed"],
                  "prompt": value["prompt"], "packet_file": packet["file"], "wire_bytes": packet["bytes"], "packet_sha256": packet["sha256"],
                  "generator_revision": model_manifest["revision"], "protocol_sha256": sha(PROTOCOL), "steps": steps, "width": plan["width"], "height": plan["height"],
                  "receiver_input": "Text packet + shared pretrained models + public CPU noise seed; no original pixels, thumbnail or image-conditioned states"}
        guard()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            noise = public_noise(value["seed"], (1, 3, value["height"], value["width"]))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = generate(model, encoder, [value["prompt"]], height=value["height"], width=value["width"],
                                  steps=steps, order=plan["order"], cfg_scale=plan["cfg_scale"], cfg_interval=tuple(plan["cfg_interval"]),
                                  shift=plan["shift"], negative_prompt=plan["negative_prompt"], noise=noise, device="cuda",
                                  num_train_timesteps=model_config.flow.num_train_timesteps, prediction=model_config.flow.prediction)
            torch.cuda.synchronize()
            if not torch.isfinite(output).all().item():
                raise ValueError("Nonfinite generator output")
            relative = Path("generated" + suffix) / f"{row['model']}-{row['budget']}-{row['case_id']}-s{value['seed']}.png"
            target = RUN / relative
            target.parent.mkdir(exist_ok=True)
            image = tensor_image(output)
            image.save(target)
            preview = RUN / ("evaluated" + suffix) / target.name
            preview.parent.mkdir(exist_ok=True)
            image.resize((plan["evaluation_width"], plan["evaluation_height"]), Image.Resampling.LANCZOS).save(preview)
            record.update({"status": "completed", "seconds": time.perf_counter() - start, "file": preview.relative_to(RUN).as_posix(), "sha256": sha(preview),
                           "native_file": relative.as_posix(), "native_sha256": sha(target), "evaluation_size": [plan["evaluation_width"], plan["evaluation_height"]], "gpu": guard()})
            del output, noise
        except Exception as exc:
            record.update({"status": "failed", "seconds": time.perf_counter() - start, "error": str(exc), "traceback": traceback.format_exc()})
            records.append(record)
            save(destination, records)
            raise
        records.append(record)
        save(destination, records)
        save(RUN / ("generation-status" + suffix + ".json"), {"phase": "generating", "at": now(), "records": len(records), "expected": len(items), "latest": {k: record[k] for k in ["model", "case_id", "budget", "seed", "seconds"]}})
        print("GENERATED", len(records), "/", len(items), row["model"], row["budget"], row["case_id"], value["seed"], round(record["seconds"], 2), flush=True)
    print("GENERATION_COMPLETE", len(records), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refine", action="store_true")
    main(parser.parse_args())
