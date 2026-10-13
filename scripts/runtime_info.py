"""Portable environment and CUDA inventory (no cooling-policy changes)."""
import json,platform,torch
print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},indent=2))
