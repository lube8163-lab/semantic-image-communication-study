"""E002 independent receiver: only packets and shared model, never originals."""
import argparse
import io
import json
import time
from pathlib import Path
from PIL import Image
from feature_packet import decode_packet
from research_runtime import ROOT,torch,write_json,load_model,public_noise,image_tensor,tensor_image,head_forward,gpu_snapshot

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",default="pilot-20261013")
    parser.add_argument("--precision",choices=["fp16_autocast","fp32"],default="fp16_autocast")
    args=parser.parse_args()
    run=ROOT / "results" / args.run
    output=run / "receiver"
    output.mkdir(parents=True,exist_ok=True)
    model,_,_,manifest=load_model()
    rows=[]
    for packet_path in sorted((run / "packets").glob("*.zip")):
        packet=packet_path.read_bytes()
        metadata,features,thumbnail=decode_packet(packet)
        if metadata["checkpoint_revision"] != manifest["revision"]:
            raise RuntimeError("Shared checkpoint revision mismatch")
        height,width=metadata["image_size"]
        seed,sigma=metadata["seed"],metadata["sigma"]
        noise=public_noise(seed,(1,3,height,width)).to("cuda:0")
        if thumbnail is None:
            x=sigma*noise
            side_info="shared noise only; clean image assumed zero"
        else:
            thumb=Image.open(io.BytesIO(thumbnail)).convert("RGB")
            approx=thumb.resize((width,height),Image.Resampling.BICUBIC)
            x=(1-sigma)*image_tensor(approx).to("cuda:0")+sigma*noise
            side_info="transmitted 64x64 JPEG and shared noise"
            approx.save(output / (packet_path.stem+"_thumbnail_only.png"))
        start=time.perf_counter()
        velocity=head_forward(model,features.to("cuda:0"),x,sigma,args.precision)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("Non-finite decoder output")
        prediction=tensor_image(x-sigma*velocity)
        name=packet_path.stem+".png"
        prediction.save(output / name)
        row={"packet":packet_path.name,"output":name,"packet_bytes":len(packet),"receiver_seconds":seconds,
             "side_information":side_info,"original_image_access":False,"precision":args.precision,
             "telemetry":gpu_snapshot()}
        rows.append(row)
        write_json(run / "receiver-progress.json",{"rows":rows})
        print(json.dumps({"packet":packet_path.name,"bytes":len(packet),"seconds":seconds}),flush=True)
    write_json(run / "receiver-summary.json",{"experiment":"E002 independent receiver","model_revision":manifest["revision"],
               "original_image_access":False,"input_paths":["packets/*.zip","shared model checkpoint"],
               "rows":rows,"scope":"Frozen-head pilot, not a trained semantic codec"})
    write_json(run / "status.json",{"phase":"receiver_complete","packets":len(rows)})

if __name__=="__main__":
    main()
