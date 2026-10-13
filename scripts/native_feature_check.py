"""Same-resolution follow-up to the initial frozen-head diagnostic; separate sender/receiver."""
import argparse
import io
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SC_FEATURE_GPU", "0"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from PIL import Image
from safetensors.torch import load_file
from feature_packet import encode_packet, decode_packet
from research_runtime import load_model, model_forward, head_forward, public_noise, image_tensor, tensor_image, gpu_snapshot
from text_route_common import ROOT, RUN, cases, read, save, sha, now, encoding_key

FOLDER = RUN / "native-feature-check"

def sender():
    model, _, _, manifest = load_model()
    null = {k: v.to("cuda") for k, v in load_file(RUN / "text-embeddings" / (encoding_key("") + ".safetensors")).items()}
    (FOLDER / "packets").mkdir(parents=True, exist_ok=True)
    rows = []
    for case in [r for r in cases() if r["id"] in ["astronaut", "coffee", "chelsea"]]:
        if sha(ROOT / case["file"]) != case["sha256"]:
            raise ValueError("Source changed")
        image = Image.open(ROOT / case["file"]).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        clean = image_tensor(image).to("cuda")
        noise = public_noise(20261013, tuple(clean.shape)).to("cuda")
        noisy = 0.9 * clean + 0.1 * noise
        output, seconds = model_forward(model, noisy, 0.1, null, [24], "fp16_autocast")
        replay = head_forward(model, output.features[24], noisy, 0.1, "fp16_autocast")
        error = float((replay - output.x.float()).abs().max())
        if error != 0:
            raise ValueError("Oracle head replay mismatch")
        feature = output.features[24][0].detach().float().cpu().numpy()
        thumbnail = io.BytesIO()
        image.resize((64, 64), Image.Resampling.LANCZOS).save(thumbnail, format="JPEG", quality=85)
        for name, bits, side, thumb in [("full_fp16_none", 16, 64, b""), ("pool4_int4_none", 4, 4, b""), ("pool4_int4_thumb64", 4, 4, thumbnail.getvalue())]:
            payload, details = encode_packet(feature, (64, 64), bits, side, 0.1, 20261013, manifest["revision"], thumb)
            path = FOLDER / "packets" / (case["id"] + "-" + name + ".zip")
            path.write_bytes(payload)
            rows.append({"case_id": case["id"], "variant": name, "packet": path.name, "sha256": sha(path), "source_sha256": case["sha256"],
                         "wire_bytes": len(payload), "components": details["uncompressed_component_bytes"], "oracle_velocity_max_error": error,
                         "sender_seconds": seconds, "scope": "Sender source is the exact 256px common input upsampled to 1024; unconditioned native forward at sigma .1, no learned decoder", "gpu": gpu_snapshot()})
            print("NATIVE_FEATURE_SENT", case["id"], name, len(payload), flush=True)
        del output, replay, clean, noisy, noise
    save(FOLDER / "sender.json", {"at": now(), "rows": rows, "oracle_role": "Original noisy RGB access only in sender replay diagnostic; excluded from communication results"})

def receiver():
    model, _, _, manifest = load_model()
    (FOLDER / "decoded").mkdir(parents=True, exist_ok=True)
    rows = []
    # Receiver reads packets and shared pretrained weights only, no source files.
    for path in sorted((FOLDER / "packets").glob("*.zip")):
        payload = path.read_bytes()
        metadata, feature, thumb = decode_packet(payload)
        if metadata["checkpoint_revision"] != manifest["revision"]:
            raise ValueError("Wrong shared model")
        case_id, variant = path.stem.split("-", 1)
        noise = public_noise(metadata["seed"], (1, 3, *metadata["image_size"])).to("cuda")
        if thumb:
            approx = Image.open(io.BytesIO(thumb)).convert("RGB").resize((1024, 1024), Image.Resampling.BICUBIC)
            noisy = 0.9 * image_tensor(approx).to("cuda") + 0.1 * noise
            thumbnail_path = FOLDER / "decoded" / (case_id + "-thumbnail64.png")
            approx.resize((256, 256), Image.Resampling.LANCZOS).save(thumbnail_path)
            rows.append({"case_id": case_id, "variant": "thumbnail64", "wire_bytes": len(thumb), "file": thumbnail_path.relative_to(RUN).as_posix(),
                         "sha256": sha(thumbnail_path), "seed": 20261013, "receiver_input": "JPEG file + shared bicubic resizing only"})
        else:
            noisy = 0.1 * noise
        modes = ["actual", "zero", "shuffle"] if thumb else ["actual"]
        for mode in modes:
            value = feature.clone()
            if mode == "zero":
                value.zero_()
            elif mode == "shuffle":
                indices = torch.randperm(value.shape[1], generator=torch.Generator(device="cpu").manual_seed(20261014))
                value = value[:, indices]
            start = time.perf_counter()
            velocity = head_forward(model, value.to("cuda"), noisy, 0.1, "fp16_autocast")
            torch.cuda.synchronize()
            if not torch.isfinite(velocity).all().item():
                raise ValueError("Nonfinite head output")
            image = tensor_image(noisy - 0.1 * velocity)
            destination = FOLDER / "decoded" / (path.stem + "-" + mode + ".png")
            image.resize((256, 256), Image.Resampling.LANCZOS).save(destination)
            rows.append({"case_id": case_id, "variant": variant if mode == "actual" else variant + "_" + mode, "packet": path.name, "wire_bytes": len(payload),
                         "file": destination.relative_to(RUN).as_posix(), "sha256": sha(destination), "seed": metadata["seed"],
                         "seconds": time.perf_counter() - start, "original_image_access": False, "decoder_mode": mode,
                         "receiver_input": "ZIP feature packet, shared weights, public noise; optional transmitted JPEG only", "gpu": gpu_snapshot()})
            save(FOLDER / "receiver.json", {"at": now(), "rows": rows, "source_image_access": False})
            print("NATIVE_FEATURE_RECEIVED", case_id, variant, mode, flush=True)
            del velocity, value
        del noise, noisy, feature
    save(FOLDER / "receiver.json", {"at": now(), "rows": rows, "source_image_access": False,
         "scope": "Matched 1024px head diagnostic on the three original photographs, source target and outputs evaluated at256. Fixed decoder zero/shuffle ablations keep the identical transmitted packet bytes; they intentionally ignore/permute features. Different compute/precision from text sampler, no learned feature adapter."})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["sender", "receiver"], required=True)
    args = p.parse_args()
    sender() if args.phase == "sender" else receiver()
