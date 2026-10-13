"""Real image-codec bytes at each actual text packet ceiling, CPU only."""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from research_runtime import psnr
from text_route_common import ROOT, RUN, all_captions, cases, read, save, sha, now

def main():
    captions = all_captions()
    folder = RUN / "codec-baselines"
    folder.mkdir(exist_ok=True)
    rows = []
    resolutions = [256, 128, 96, 64, 48, 32, 24, 16, 8]
    qualities = [0, 5, 10, 20, 40, 60, 80, 95]
    for case in cases():
        image = Image.open(ROOT / case["file"]).convert("RGB")
        candidates = []
        for fmt in ["JPEG", "WEBP"]:
            for size in resolutions:
                small = image.resize((size, size), Image.Resampling.LANCZOS)
                for quality in qualities:
                    buffer = io.BytesIO()
                    options = {"optimize": True, "subsampling": 2} if fmt == "JPEG" else {"method": 6}
                    small.save(buffer, format=fmt, quality=max(1, quality) if fmt == "JPEG" else quality, **options)
                    payload = buffer.getvalue()
                    decoded = Image.open(io.BytesIO(payload)).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
                    candidates.append({"format": fmt, "encoded_size": size, "quality": quality, "bytes": len(payload),
                                       "psnr": psnr(image, decoded), "payload": payload, "decoded": decoded})
        for caption in [r for r in captions if r["case_id"] == case["id"] and r["status"] == "completed"]:
            for fmt in ["JPEG", "WEBP"]:
                for mode in ["fixed256", "adaptive_resolution"]:
                    eligible = [r for r in candidates if r["format"] == fmt and r["bytes"] <= caption["wire_bytes"] and
                                (mode != "fixed256" or r["encoded_size"] == 256)]
                    row = {"at": now(), "case_id": case["id"], "sender_model": caption["model"], "caption_budget": caption["budget"],
                           "format": fmt, "mode": mode, "ceiling_bytes": caption["wire_bytes"],
                           "protocol": "Image-file bytes include container and dimensions; shared decoder resizes output to 256px. No additional per-image side information."}
                    if not eligible:
                        row.update({"status": "not_fit", "minimum_candidate_bytes": min(r["bytes"] for r in candidates if r["format"] == fmt and (mode != "fixed256" or r["encoded_size"] == 256))})
                    else:
                        best = max(eligible, key=lambda r: float("inf") if r["psnr"] is None else r["psnr"])
                        name = f"{caption['model']}-{caption['budget']}-{case['id']}-{fmt.lower()}-{mode}"
                        encoded = folder / (name + (".jpg" if fmt == "JPEG" else ".webp"))
                        decoded = folder / (name + ".png")
                        encoded.write_bytes(best["payload"])
                        best["decoded"].save(decoded)
                        row.update({"status": "completed", "wire_bytes": encoded.stat().st_size, "encoded_size": best["encoded_size"], "quality": best["quality"],
                                    "encoded_file": encoded.relative_to(RUN).as_posix(), "decoded_file": decoded.relative_to(RUN).as_posix(),
                                    "encoded_sha256": sha(encoded), "decoded_sha256": sha(decoded), "psnr_db": best["psnr"]})
                    rows.append(row)
        save(RUN / "codec-baselines.json", rows)
        print("CODEC_BASELINES", case["id"], len(rows), flush=True)
    save(RUN / "codec-protocol.json", {"at": now(), "role": "Secondary source-fidelity diagnostic; added before regeneration, does not change caption/generator arms",
         "resolutions": resolutions, "qualities": qualities, "selection": "Highest source PSNR among finite candidate grid fitting the complete actual text-packet byte ceiling, chosen by sender; no oracle source access at receiver",
         "primary_mode": "fixed256", "additional_mode": "adaptive_resolution, to avoid confusing a 256px encoding-size failure with a codec's ability to send a small thumbnail",
         "records": len(rows), "completed": sum(r["status"] == "completed" for r in rows), "not_fit": sum(r["status"] == "not_fit" for r in rows)})

if __name__ == "__main__":
    main()
