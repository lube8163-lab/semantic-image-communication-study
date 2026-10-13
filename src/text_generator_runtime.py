"""Native-resolution Iris inference on the RTX, with read-only source weights."""
import subprocess
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from iris3b.config import inference_config
from iris3b.models.dit import IrisDiT
from text_route_common import ROOT, read, now

RTX_UUID = os.environ.get("SC_GENERATOR_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]

def guard():
    from gpu_monitor import check
    return {"at": now(), **check("generator"), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

def load_iris():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A BF16-capable CUDA GPU is required for this generator")
    guard()
    config = inference_config(OmegaConf.to_container(OmegaConf.load(ROOT / "models/iris-3b/config.yaml")), ["model.attn_backend=sdpa"])
    with torch.device("meta"):
        model = IrisDiT(config.model)
    weights = load_file(ROOT / "models/iris-3b/model.safetensors")
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.eval().requires_grad_(False).to("cuda", dtype=torch.bfloat16)
    backend = os.environ.get("IRIS_ATTENTION_BACKEND", "sdpa")
    # The 300-token text refiner has an explicit padding mask; preserve it on
    # standard SDPA. Only the unmasked image/text trunk and pixel head use the
    # explicitly selected fused backend. No weights, masks or equations change.
    for name, module in model.named_modules():
        if hasattr(module, "backend") and not name.startswith("y_embedder"):
            module.backend = backend
    return model, config, read(ROOT / "sources/model-manifest.json")
