"""E001 observation + E002 sender. Receiver is a separate command/process."""
import argparse
import io
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from feature_packet import encode_packet
from research_runtime import ROOT,torch,write_json,load_model,public_noise,image_tensor,tensor_image,model_forward,head_forward,gpu_snapshot,psnr,mae

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",default="pilot-20261013")
    parser.add_argument("--limit",type=int,default=5)
    parser.add_argument("--precision",choices=["auto","fp16_autocast","fp32"],default="auto")
    args=parser.parse_args()
    run=ROOT / "results" / args.run
    if (run / "encoder-summary.json").exists():
        raise RuntimeError("Completed run exists; use a new --run")
    for folder in ["features","packets","inputs","diagnostics"]:
        (run / folder).mkdir(parents=True,exist_ok=True)
    write_json(run / "status.json",{"phase":"loading_model","at":time.time()})
    model,prompt,config,manifest=load_model()
    samples=json.loads((ROOT / "data/manifest.json").read_text())["images"][:args.limit]
    sigmas=[0.1,0.5,0.9]
    layers=[1,8,10,16,24]
    rows,packets,oracle,failures=[],[],[],[]
    precision=args.precision
    for i,sample in enumerate(samples):
        image=Image.open(ROOT / sample["file"]).convert("RGB")
        image.save(run / "inputs" / (sample["id"]+".png"))
        x0=image_tensor(image).to("cuda:0")
        seed=20261013
        noise=public_noise(seed,x0.shape).to("cuda:0")
        for sigma in sigmas:
            x=(1-sigma)*x0+sigma*noise
            if precision == "auto":
                precision="fp16_autocast"
            write_json(run / "status.json",{"phase":"extracting","image":sample["id"],"sigma":sigma,"precision":precision})
            out,seconds=model_forward(model,x,sigma,prompt,layers,precision)
            if not bool(torch.isfinite(out.x).all()) or not all(bool(torch.isfinite(f).all()) for f in out.features.values()):
                failures.append({"image":sample["id"],"sigma":sigma,"precision":precision,"failure":"non-finite tensors"})
                if args.precision != "auto" or precision == "fp32":
                    raise RuntimeError("Non-finite model output")
                precision="fp32"
                out,seconds=model_forward(model,x,sigma,prompt,layers,precision)
                if not bool(torch.isfinite(out.x).all()) or not all(bool(torch.isfinite(f).all()) for f in out.features.values()):
                    raise RuntimeError("Non-finite model output even in FP32")
            diagnostic=tensor_image(x-sigma*out.x.float())
            name=f"{sample['id']}_s{sigma:.1f}"
            diagnostic.save(run / "diagnostics" / (name+"_single_step.png"))
            feature_path=run / "features" / (name+".npz")
            arrays={f"layer{layer}":out.features[layer][0].detach().float().cpu().numpy() for layer in layers}
            np.savez_compressed(feature_path,**arrays)
            summaries={str(layer):{"shape":list(arrays[f"layer{layer}"].shape),
                         "min":float(arrays[f"layer{layer}"].min()),"max":float(arrays[f"layer{layer}"].max()),
                         "mean":float(arrays[f"layer{layer}"].mean()),"std":float(arrays[f"layer{layer}"].std()),
                         "fp32_tensor_bytes":int(arrays[f"layer{layer}"].nbytes),
                         "fp16_tensor_bytes":int(arrays[f"layer{layer}"].size*2),
                         "finite":bool(np.isfinite(arrays[f"layer{layer}"]).all())} for layer in layers}
            row={"image":sample["id"],"kind":sample["kind"],"input_sha256":sample["sha256"],
                 "sigma":sigma,"seed":seed,"text_condition":"official cached empty prompt",
                 "precision":precision,"layers":summaries,"forward_seconds":seconds,
                 "feature_archive_bytes":feature_path.stat().st_size,"diagnostic_psnr":psnr(image,diagnostic),
                 "diagnostic_mae":mae(image,diagnostic),"telemetry":gpu_snapshot()}
            rows.append(row)
            print(json.dumps({"image":sample["id"],"sigma":sigma,"seconds":seconds,
                              "precision":precision,"peak_GiB":row["telemetry"]["torch_peak_allocated_bytes"]/2**30}),flush=True)
            if sigma == 0.1:
                # Exact head replay uses original x: a validation control, NEVER a codec.
                replay=head_forward(model,out.features[24],x,sigma,precision)
                error=float((replay-out.x.float()).abs().max())
                if error > 0.05:
                    raise RuntimeError(f"Head split validation failed: max velocity error {error}")
                oracle.append({"image":sample["id"],"max_velocity_error":error,"original_pixels_available":True})
                thumb_stream=io.BytesIO()
                image.resize((64,64),Image.Resampling.LANCZOS).save(thumb_stream,format="JPEG",quality=85,optimize=True,subsampling=0)
                thumbnail=thumb_stream.getvalue()
                variants=[("full_fp16_none",16,16,False),("full_int8_none",8,16,False),
                          ("pool4_int4_none",4,4,False),("full_fp16_thumb64",16,16,True),
                          ("pool4_int4_thumb64",4,4,True),("pool2_int4_thumb64",4,2,True)]
                for variant,bits,pool_side,use_thumb in variants:
                    packet,account=encode_packet(arrays["layer24"],(16,16),bits,pool_side,sigma,seed,
                                                 manifest["revision"],thumbnail if use_thumb else b"")
                    filename=name+"_"+variant+".zip"
                    (run / "packets" / filename).write_bytes(packet)
                    packets.append({"image":sample["id"],"variant":variant,"file":filename,
                                    "accounting":account,"encoder_precision":precision})
            write_json(run / "encoder-progress.json",{"rows":rows,"packets":packets,"oracle_replay":oracle,"failures":failures})
            del out
    summary={"experiment":"E001/E002 sender","model":manifest,"config":config.model.__dict__ | {"pixel":config.model.pixel.__dict__},
             "resolution":[256,256],"sigmas":sigmas,"layers":layers,"rows":rows,
             "packets":packets,"oracle_replay":oracle,"failures":failures,
             "prompt_note":"Empty-prompt states exported with upscaler reused with matching base architecture. All 300 token positions retained."}
    write_json(run / "encoder-summary.json",summary)
    write_json(run / "status.json",{"phase":"sender_complete","packets":len(packets),"forwards":len(rows)})
    print("Sender complete",flush=True)

if __name__=="__main__":
    main()
