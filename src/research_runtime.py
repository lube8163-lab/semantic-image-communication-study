"""Project-local loading, public noise and monitored single-GPU inference."""
import json
import os
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SC_FEATURE_GPU", "0"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from iris3b.config import inference_config
from iris3b.models.dit import IrisDiT

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = os.environ.get("SC_FEATURE_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def cooling_guard():
    from gpu_monitor import check
    return check("feature")

def gpu_snapshot():
    line = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
        "--query-gpu=uuid,name,memory.used,temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    return {"at":datetime.now(timezone.utc).isoformat(),"nvidia_smi":line,
            "torch_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"cooling":cooling_guard()}

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    cooling_guard()
    config = inference_config(OmegaConf.to_container(OmegaConf.load(ROOT / "models/iris-3b/config.yaml")), ["model.attn_backend=sdpa"])
    with torch.device("meta"):
        model = IrisDiT(config.model)
    weights = load_file(ROOT / "models/iris-3b/model.safetensors")
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.eval().requires_grad_(False).to("cuda:0", torch.float32)
    prompt = load_file(ROOT / "models/iris-3b/upscaler/empty_prompt.safetensors")
    prompt = {k:v.to("cuda:0") for k,v in prompt.items()}
    manifest = json.loads((ROOT / "sources/model-manifest.json").read_text(encoding="utf-8"))
    return model, prompt, config, manifest

def public_noise(seed, shape):
    return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)

def image_tensor(image):
    array = np.asarray(image.convert("RGB")).copy()
    return torch.from_numpy(array).permute(2,0,1)[None].float() / 127.5 - 1

def tensor_image(tensor):
    array = ((tensor[0].detach().float().cpu().clamp(-1,1).permute(1,2,0).numpy() + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(array)

def precision_context(precision):
    return torch.autocast("cuda", dtype=torch.float16) if precision == "fp16_autocast" else nullcontext()

def model_forward(model, x, sigma, prompt, capture, precision):
    cooling_guard()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), precision_context(precision):
        t = torch.tensor([sigma*1000], device=x.device, dtype=torch.float32)
        out = model(x, t, prompt["embeddings"], capture=tuple(capture), y_mask=prompt["mask"])
    torch.cuda.synchronize()
    return out, time.perf_counter() - start

def head_forward(model, features, x, sigma, precision):
    """Run only the existing final PiT head; x is an explicit required input."""
    cooling_guard()
    cfg = model.cfg
    p = cfg.patch_size
    batch, _, height, width = x.shape
    grid = (height//p,width//p)
    patches = grid[0] * grid[1]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), precision_context(precision):
        t = torch.tensor([sigma*1000],device=x.device,dtype=torch.float32)
        s = F.silu(model.t_embedder(t) + features)
        pixels = model.pixel_embedder(x)
        rope = model._fetch_rope_pix(grid,x.device)
        for block in model.pixel_blocks:
            pixels = block(pixels,s.reshape(batch*patches,-1),rope,grid)
        out = model.final_layer(pixels)
        folded = out.reshape(batch,patches,p*p,-1).permute(0,3,2,1).reshape(batch,-1,patches)
        velocity = F.fold(folded,output_size=(height,width),kernel_size=p,stride=p)
    return velocity.float()

def psnr(a,b):
    mse = float(np.mean((np.asarray(a).astype(np.float64)-np.asarray(b).astype(np.float64))**2))
    return None if mse == 0 else float(10*np.log10(255**2/mse))

def mae(a,b):
    return float(np.mean(np.abs(np.asarray(a).astype(np.float64)-np.asarray(b).astype(np.float64))))
