"""Independent CLIP/DINO image similarity and separate pixel distortion metrics."""
import os
import sys
import zipfile
import argparse
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SC_FEATURE_GPU", "0"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPConfig, CLIPVisionModelWithProjection, CLIPImageProcessor, AutoImageProcessor, AutoModel
from research_runtime import cooling_guard, gpu_snapshot, psnr, mae
from text_route_common import ROOT, RUN, protocol, cases, read, save, sha, now

CLIP_DIR = Path(os.environ.get("SC_CLIP_DIR", str(ROOT / "models/clip-evaluator")))
DINO_DIR = Path(os.environ.get("SC_DINO_DIR", str(ROOT / "models/dino")))

def load_evaluators():
    cfg = CLIPConfig.from_pretrained(CLIP_DIR / "safety_checker", local_files_only=True)
    cfg.vision_config.projection_dim = cfg.projection_dim
    clip = CLIPVisionModelWithProjection(cfg.vision_config)
    weights = {}
    position_buffer_verified = False
    with safe_open(CLIP_DIR / "safety_checker/model.fp16.safetensors", framework="pt") as file:
        for name in file.keys():
            if name.startswith("vision_model.vision_model."):
                value = file.get_tensor(name)
                if name.endswith("embeddings.position_ids"):
                    if not torch.equal(value.long(), clip.vision_model.embeddings.position_ids):
                        raise ValueError("CLIP position buffer differs from native deterministic positions")
                    position_buffer_verified = True
                else:
                    weights[name[len("vision_model."):]] = value.float()
            elif name.startswith("visual_projection."):
                weights[name] = file.get_tensor(name).float()
    clip.load_state_dict(weights, strict=True)
    clip.to("cuda").eval().requires_grad_(False)
    cp = CLIPImageProcessor.from_pretrained(CLIP_DIR / "feature_extractor", local_files_only=True)
    dino = AutoModel.from_pretrained(DINO_DIR, local_files_only=True, dtype=torch.float32).to("cuda").eval().requires_grad_(False)
    dp = AutoImageProcessor.from_pretrained(DINO_DIR, local_files_only=True, use_fast=False)
    save(RUN / "evaluator-provenance.json", {"at": now(), "clip": {"source": "CompVis/stable-diffusion-safety-checker CLIP ViT-L/14 visual encoder and projection only",
         "local_path": str(CLIP_DIR / "safety_checker"), "checkpoint_sha256": sha(CLIP_DIR / "safety_checker/model.fp16.safetensors"),
         "load": "Exact visual/projection parameters mapped to native CLIPVisionModelWithProjection with strict=True; nonpersistent deterministic position_ids equality checked separately", "position_buffer_verified": position_buffer_verified, "precision": "float32", "projection_dim": cfg.projection_dim},
         "dino": read(ROOT / "sources/text-route-cached-assets.json")["dino"], "limitation": "DINO-family teacher overlaps Iris REPA design; CLIP corroboration and visual review are needed. Image cosines do not certify identity, counts or geometry.", "gpu": gpu_snapshot()})
    def embed(images):
        cooling_guard()
        with torch.inference_mode():
            c = clip(cp(images=images, return_tensors="pt").pixel_values.to("cuda")).image_embeds
            d = dino(**dp(images=images, return_tensors="pt").to("cuda")).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(c, dim=-1).cpu().numpy(), torch.nn.functional.normalize(d, dim=-1).cpu().numpy()
    return embed

def measure():
    sources = {r["id"]: r for r in cases()}
    images = {k: Image.open(ROOT / r["file"]).convert("RGB") for k, r in sources.items()}
    if any(sha(ROOT / r["file"]) != r["sha256"] for r in sources.values()):
        raise ValueError("Evaluator source changed")
    embed = load_evaluators()
    source_c, source_d = embed(list(images.values()))
    source_embeddings = {k: (source_c[i], source_d[i]) for i, k in enumerate(images)}
    items = []
    for r in read(RUN / "generation.json") if (RUN / "generation.json").exists() else []:
        if r["status"] != "completed":
            continue
        if r["model"] == "empty_control":
            for case_id in sources:
                items.append({**r, "case_id": case_id, "route": "empty_control", "path": str(RUN / r["file"])})
        else:
            items.append({**r, "route": "text_only", "path": str(RUN / r["file"])})
    refinement = RUN / "generation-refine40.json"
    if refinement.exists():
        for r in read(refinement):
            if r["status"] == "completed":
                items.append({**r, "route": "text_refine40", "path": str(RUN / r["file"])})
    pilot = ROOT / "results/pilot-20261013"
    for r in read(pilot / "receiver-summary.json")["rows"]:
        case_id = next(k for k in sources if r["output"].startswith(k + "_s0.1_"))
        items.append({"route": "initial_feature_head", "model": "Iris_frozen_head", "budget": r["output"].split("_s0.1_", 1)[1].removesuffix(".png"),
                      "case_id": case_id, "seed": 20261013, "wire_bytes": r["packet_bytes"], "path": str(pilot / "receiver" / r["output"]), "file": r["output"]})
    for case_id in [r["id"] for r in cases() if r["cohort"] == "initial_pilot"]:
        path = pilot / "receiver" / (case_id + "_s0.1_pool4_int4_thumb64_thumbnail_only.png")
        if path.exists():
            packet = pilot / "packets" / (case_id + "_s0.1_pool4_int4_thumb64.zip")
            if packet.exists():
                with zipfile.ZipFile(packet) as archive:
                    nbytes = len(archive.read("thumbnail.jpg"))
            else:
                source = next(r for r in read(pilot / "encoder-summary.json")["packets"] if r["image"] == case_id and r["variant"] == "pool4_int4_thumb64")
                nbytes = source["accounting"]["uncompressed_component_bytes"]["thumbnail.jpg"]
            items.append({"route": "thumbnail", "model": "thumbnail64", "budget": "thumbnail", "case_id": case_id, "seed": 20261013,
                          "wire_bytes": nbytes, "path": str(path), "file": path.name})
    for r in read(pilot / "evaluation-summary.json")["comparison_rows"]:
        for fmt in ["jpeg", "webp"]:
            entry = r[fmt]
            items.append({"route": "initial_codec", "model": fmt.upper(), "budget": r["variant"], "case_id": r["image"],
                          "seed": None, "wire_bytes": entry["bytes"], "path": str(pilot / "evaluation" / entry["file"]), "file": entry["file"]})
    native_features = RUN / "native-feature-check/receiver.json"
    if native_features.exists():
        for r in read(native_features)["rows"]:
            items.append({**r, "route": "native_feature_head" if r["variant"] != "thumbnail64" else "native_thumbnail",
                          "model": "Iris_frozen_head" if r["variant"] != "thumbnail64" else "thumbnail64", "budget": r["variant"], "path": str(RUN / r["file"])})
    baselines = RUN / "codec-baselines.json"
    if baselines.exists():
        for r in read(baselines):
            if r["status"] == "completed":
                items.append({**r, "route": "codec", "model": r["format"], "budget": r["mode"], "seed": None, "path": str(RUN / r["decoded_file"]), "file": r["decoded_file"]})
                items[-1]["sha256"] = r["decoded_sha256"]
    metrics = read(RUN / "metrics.json") if (RUN / "metrics.json").exists() else []
    def identity(r):
        return tuple(r.get(k) for k in ["route", "model", "budget", "case_id", "seed", "path"])
    done = {identity(r) for r in metrics}
    items = [r for r in items if identity(r) not in done]
    for start in range(0, len(items), 12):
        batch = items[start:start + 12]
        ims = []
        for item in batch:
            if item.get("sha256") and sha(item["path"]) != item["sha256"]:
                raise ValueError("Generated image changed")
            im = Image.open(item["path"]).convert("RGB")
            if im.size != (256, 256):
                raise ValueError("Unexpected output size")
            ims.append(im)
        c, d = embed(ims)
        for i, (item, im) in enumerate(zip(batch, ims)):
            sc, sd = source_embeddings[item["case_id"]]
            r = {k: v for k, v in item.items() if k not in ["gpu", "traceback", "prompt"]}
            r.update({"kind": sources[item["case_id"]]["kind"], "category": sources[item["case_id"]]["category"],
                      "clip_image_cosine": float(np.dot(c[i], sc)), "dino_image_cosine": float(np.dot(d[i], sd)),
                      "psnr_db": psnr(images[item["case_id"]], im), "mae_rgb8": mae(images[item["case_id"]], im)})
            metrics.append(r)
        save(RUN / "metrics.json", metrics)
        print("SCORED", len(metrics), "pending_batch", start + len(batch), "/", len(items), flush=True)
    return metrics

def aggregates(metrics):
    groups = defaultdict(list)
    for row in metrics:
        if row["kind"] == "photograph" and row["route"] != "codec":
            groups[(row["route"], row["model"], row["budget"])].append(row)
    summary = []
    rng = np.random.default_rng(20261013)
    keys = ["clip_image_cosine", "dino_image_cosine", "psnr_db", "mae_rgb8", "wire_bytes"]
    cells = {}
    for group, rows in sorted(groups.items()):
        per_image = defaultdict(list)
        for r in rows:
            per_image[r["case_id"]].append(r)
        averages = {k: {metric: float(np.mean([r[metric] for r in rs if r.get(metric) is not None]))
                        for metric in keys if any(r.get(metric) is not None for r in rs)} for k, rs in per_image.items()}
        cells[group] = averages
        value = {"route": group[0], "model": group[1], "budget": group[2], "photos": len(averages), "generated_observations": len(rows)}
        for metric in keys:
            array = np.array([r[metric] for r in averages.values() if metric in r])
            if len(array):
                value[metric] = float(array.mean())
                boot = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
                value[metric + "_image_bootstrap95"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        summary.append(value)
    comparisons = []
    pairs = []
    for budget in ["compact", "detailed"]:
        for other in ["minicpm", "lfm16"]:
            pairs.append((("text_only", "qwen4b", budget), ("text_only", other, budget)))
    for name in protocol()["models"]:
        pairs.append((("text_only", name, "detailed"), ("text_only", name, "compact")))
    for name in protocol()["models"]:
        pairs.append((("text_only", name, "compact"), ("empty_control", "empty_control", "empty")))
    for left, right in pairs:
        if left not in cells or right not in cells:
            continue
        ids = sorted(set(cells[left]) & set(cells[right]))
        value = {"left": list(left), "right": list(right), "paired_photos": len(ids), "seed_handling": "Mean over seeds within source image before pairing", "metrics": {}}
        for metric in keys[:4]:
            differences = np.array([cells[left][case_id][metric] - cells[right][case_id][metric] for case_id in ids])
            boot = differences[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(axis=1)
            value["metrics"][metric] = {"mean_difference": float(differences.mean()), "bootstrap95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                                        "image_wins": int((differences > 0).sum()), "image_losses": int((differences < 0).sum())}
        comparisons.append(value)
    refinement = []
    first_seed = protocol()["seeds"][0]
    for name in protocol()["models"]:
        for budget in ["compact", "detailed"]:
            left = {r["case_id"]: r for r in metrics if r["route"] == "text_refine40" and r["model"] == name and r["budget"] == budget}
            right = {r["case_id"]: r for r in metrics if r["route"] == "text_only" and r["model"] == name and r["budget"] == budget and r["seed"] == first_seed}
            ids = sorted(set(left) & set(right))
            if ids:
                value = {"model": name, "budget": budget, "paired_photos": len(ids), "seed": first_seed, "comparison": "40 steps minus10, same caption/source/seed; three-photo diagnostic only", "metrics": {}}
                for metric in keys[:4]:
                    diff = [left[k][metric] - right[k][metric] for k in ids]
                    value["metrics"][metric] = {"mean_difference": float(np.mean(diff)), "individual_differences": dict(zip(ids, diff))}
                refinement.append(value)
    ablations = []
    actual = cells.get(("native_feature_head", "Iris_frozen_head", "pool4_int4_thumb64"), {})
    for group in [("native_feature_head", "Iris_frozen_head", "pool4_int4_thumb64_zero"),
                  ("native_feature_head", "Iris_frozen_head", "pool4_int4_thumb64_shuffle"), ("native_thumbnail", "thumbnail64", "thumbnail64")]:
        comparator = cells.get(group, {})
        ids = sorted(set(actual) & set(comparator))
        if ids:
            ablations.append({"left": "Actual pooled features plus thumbnail", "right": list(group), "paired_photos": len(ids),
                              "mean_differences": {k: float(np.mean([actual[i][k] - comparator[i][k] for i in ids])) for k in keys[:4]}})
    codec_summary = []
    for name in protocol()["models"]:
        for budget in ["compact", "detailed"]:
            for fmt in ["JPEG", "WEBP"]:
                for mode in ["fixed256", "adaptive_resolution"]:
                    rows = [r for r in metrics if r["route"] == "codec" and r["kind"] == "photograph" and
                            (r["sender_model"], r["caption_budget"], r["model"], r["budget"]) == (name, budget, fmt, mode)]
                    value = {"sender_model": name, "caption_budget": budget, "format": fmt, "mode": mode, "photos_encoded": len(rows), "photos_expected": 11}
                    if rows:
                        value.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
                    codec_summary.append(value)
    save(RUN / "evaluation-summary.json", {"at": now(), "photo_aggregates": summary, "paired_comparisons": comparisons,
         "refinement_same_seed": refinement, "native_feature_ablation_comparisons": ablations,
         "same_text_ceiling_codec_aggregates": codec_summary,
         "limitations": "Exploratory 11-photo convenience cohort. Descriptive percentile bootstrap, 10000 resamples, unadjusted multiple comparisons, unknown pretraining overlap. Synthetic controls excluded. Image cosines are not exact fidelity or identity measurements. Initial feature decoder and text sampler have different byte budgets and compute."})
    print("EVALUATION_COMPLETE", len(metrics), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true", help="Cache metrics for available outputs and codecs without final aggregation")
    args = p.parse_args()
    measured = measure()
    if not args.prepare:
        aggregates(measured)
