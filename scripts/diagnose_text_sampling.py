"""Trace unclamped rectified-flow states and compare precision/resolution."""
import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SC_FEATURE_GPU", "0"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import load_file
from iris3b.flow.solver import FlowDPMSolver
from research_runtime import load_model, public_noise, tensor_image, cooling_guard, gpu_snapshot
from text_route_common import RUN, read, save, encoding_key, now
from text_generator_runtime import load_iris, guard as rtx_guard

def stats(tensor):
    tensor = tensor.float()
    return {"mean": float(tensor.mean()), "std": float(tensor.std()), "absmax": float(tensor.abs().max()), "finite": bool(torch.isfinite(tensor).all()),
            "abs_gt1_fraction": float((tensor.abs() > 1).float().mean())}

def main(args):
    if args.gpu == "rtx":
        model, cfg, _ = load_iris()
    else:
        model, _, cfg, _ = load_model()
    caption = next(r["prompt"] for r in read(RUN / "captions/qwen4b.json") if r["case_id"] == "astronaut" and r["budget"] == "compact")
    positive = load_file(RUN / "text-embeddings" / (encoding_key(caption) + ".safetensors"))
    null = load_file(RUN / "text-embeddings" / (encoding_key("") + ".safetensors"))
    dtype = next(model.parameters()).dtype
    cond = positive["embeddings"].to("cuda", dtype=dtype)
    uncond = null["embeddings"].to("cuda", dtype=dtype)
    cmask = positive["mask"].to("cuda")
    mask = torch.cat([null["mask"], positive["mask"]]).to("cuda")
    trace = []
    def fn(x, t, y):
        rtx_guard() if args.gpu == "rtx" else cooling_guard()
        start = time.perf_counter()
        result = model(x, t, y, y_mask=cmask if y is cond else mask).x.float()
        torch.cuda.synchronize()
        entry = {"step": len(trace) + 1, "sigma": float(t[0]) / 1000, "input": stats(x), "velocity": stats(result), "seconds": time.perf_counter() - start}
        trace.append(entry)
        print(entry, flush=True)
        return result
    precision = torch.autocast("cuda", dtype=getattr(torch, "bfloat16" if args.precision == "bf16" else "float16")) if args.precision != "fp32" else nullcontext()
    solver = FlowDPMSolver(fn, cfg_scale=3, prediction="v")
    noise = public_noise(20261013, (1, 3, args.size, args.size)).to("cuda")
    start = time.perf_counter()
    attention = nullcontext() if args.gpu == "rtx" else sdpa_kernel(SDPBackend.MATH)
    with torch.inference_mode(), attention, precision:
        output = solver.sample(noise, cond, uncond, steps=args.steps, order=2, shift=4)
    label = f"{args.precision}-{args.size}-{args.steps}"
    tensor_image(output).save(RUN / ("diagnostic-" + label + ".png"))
    save(RUN / ("diagnostic-" + label + ".json"), {"at": now(), "precision": args.precision, "size": args.size, "steps": args.steps,
         "prompt": caption, "seed": 20261013, "trace": trace, "unclamped_output": stats(output), "seconds": time.perf_counter() - start,
         "gpu": rtx_guard() if args.gpu == "rtx" else gpu_snapshot()})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=["fp16", "fp32", "bf16"], required=True)
    p.add_argument("--gpu", choices=["v100", "rtx"], default="v100")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=10)
    main(p.parse_args())
