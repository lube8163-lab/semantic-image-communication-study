"""Receiver-local text-only states from packets; no source-image access."""
import dataclasses
import os
import sys
import time
import traceback
from pathlib import Path
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SC_FEATURE_GPU", "0"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from iris3b.config import inference_config
from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
from research_runtime import cooling_guard, gpu_snapshot
from text_route_common import ROOT, RUN, MODEL_PATHS, PROTOCOL, protocol, all_captions, read, save, sha, now, wire, encoding_key

def main():
    plan = protocol()
    rows = all_captions()
    cache = RUN / "text-embeddings"
    packets = RUN / "text-packets"
    cache.mkdir(exist_ok=True)
    packets.mkdir(exist_ok=True)
    cooling_guard()
    config = inference_config(OmegaConf.to_container(OmegaConf.load(ROOT / "models/iris-3b/config.yaml")))
    cfg = dataclasses.replace(config.text_encoder, pretrained=str(MODEL_PATHS["qwen4b"]), dtype="float16", compile=False,
                              on_caption_overflow="error", null_embed_dir=str(cache / "null"))
    start = time.perf_counter()
    encoder = Qwen3VLTextEncoder(cfg, device="cuda")
    print("TEXT_ENCODER_READY", time.perf_counter() - start, flush=True)
    records = []
    prompts = [(None, "")] + [(row, row["prompt"]) for row in rows if row["status"] == "completed"]
    for i, (row, prompt) in enumerate(prompts):
        key = encoding_key(prompt)
        target = cache / (key + ".safetensors")
        record = {"key": key, "model": row["model"] if row else "empty_control", "case_id": row["case_id"] if row else "empty_control",
                  "budget": row["budget"] if row else "empty", "prompt": prompt, "at": now(), "source": "Text only, re-encoded independently at receiver",
                  "caption_token_budget": encoder.caption_budget, "protocol_sha256": sha(PROTOCOL)}
        try:
            n = len(encoder.tokenizer.encode(prompt, add_special_tokens=False))
            record["caption_tokens"] = n
            if n > encoder.caption_budget:
                raise ValueError(f"Iris caption overflow: {n} > {encoder.caption_budget}")
            cooling_guard()
            torch.cuda.synchronize()
            start = time.perf_counter()
            record["cache_hit"] = target.exists()
            if not target.exists():
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    encoded = encoder.encode([prompt])
                torch.cuda.synchronize()
                if not torch.isfinite(encoded.embeddings).all().item():
                    raise ValueError("Nonfinite text embeddings")
                if tuple(encoded.embeddings.shape) != (1, 300, 12, 2560):
                    raise ValueError("Wrong Iris text state shape")
                save_file({"embeddings": encoded.embeddings.cpu().contiguous(), "mask": encoded.mask.cpu().contiguous()}, target,
                          metadata={"prompt_sha256": key, "qwen_revision": read(ROOT / "sources/qwen3-vl-4b-manifest.json")["revision"],
                                    "precision": "float16", "text_only": "true"})
                del encoded
            record.update({"encoding_seconds": time.perf_counter() - start, "sha256": sha(target), "file": target.relative_to(RUN).as_posix(), "status": "completed"})
            record["packets"] = []
            for seed in plan["seeds"]:
                name = f"{row['model']}-{row['budget']}-{row['case_id']}-{seed}.json" if row else f"empty-{seed}.json"
                packet = packets / name
                packet.write_bytes(wire(prompt, seed))
                record["packets"].append({"file": packet.relative_to(RUN).as_posix(), "bytes": packet.stat().st_size, "sha256": sha(packet), "seed": seed})
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        records.append(record)
        save(RUN / "text-encoding.json", records)
        print("TEXT_ENCODED", i + 1, "/", len(prompts), record["model"], record["case_id"], record["budget"], record["status"], flush=True)
    null = load_file(cache / (encoding_key("") + ".safetensors"))
    official = load_file(ROOT / "models/iris-3b/upscaler/empty_prompt.safetensors")
    difference = null["embeddings"].float() - official["embeddings"].float()
    save(RUN / "text-encoder-summary.json", {"at": now(), "successful": sum(r["status"] == "completed" for r in records),
         "failed": [r for r in records if r["status"] != "completed"], "caption_token_budget": encoder.caption_budget, "max_caption_tokens": max(r.get("caption_tokens", 0) for r in records),
         "empty_vs_supplied_empty_relative_l2": float(difference.norm() / official["embeddings"].float().norm()),
         "empty_masks_equal": torch.equal(null["mask"], official["mask"]), "null_choice": "Independently encoded empty text using the same FP16 language encoder as positive captions",
         "gpu": gpu_snapshot()})
    if any(r["status"] != "completed" for r in records):
        raise RuntimeError("Text state failures retained; do not silently omit from batch")
    print("TEXT_ENCODING_COMPLETE", len(records), flush=True)

if __name__ == "__main__":
    main()
