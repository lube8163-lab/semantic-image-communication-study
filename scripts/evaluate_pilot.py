"""Evaluate after the independent receiver exits, then build a local image report."""
import argparse
import csv
import html
import io
import json
import statistics
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from research_runtime import ROOT,write_json,psnr,mae

def budget_codec(image,fmt,budget):
    candidates=[]
    for quality in range(1,101):
        stream=io.BytesIO()
        kwargs={"quality":quality}
        if fmt=="JPEG":
            kwargs.update(optimize=True,subsampling=0)
        image.save(stream,format=fmt,**kwargs)
        payload=stream.getvalue()
        if len(payload)<=budget:
            candidates.append((quality,payload))
    if not candidates:
        return None
    quality,payload=max(candidates,key=lambda item:item[0])
    return {"quality":quality,"bytes":len(payload),"payload":payload,
            "image":Image.open(io.BytesIO(payload)).convert("RGB")}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",default="pilot-20261013")
    args=parser.parse_args()
    run=ROOT / "results" / args.run
    encoder=json.loads((run / "encoder-summary.json").read_text())
    receiver=json.loads((run / "receiver-summary.json").read_text())
    output=run / "evaluation"
    output.mkdir(exist_ok=True)
    lookup={entry["file"]:entry for entry in encoder["packets"]}
    rows=[]
    for received in receiver["rows"]:
        sent=lookup[received["packet"]]
        image_id,variant=sent["image"],sent["variant"]
        original=Image.open(run / "inputs" / (image_id+".png")).convert("RGB")
        reconstruction=Image.open(run / "receiver" / received["output"]).convert("RGB")
        budget=received["packet_bytes"]
        row={"image":image_id,"variant":variant,"packet_bytes":budget,"bpp":budget*8/(256*256),
             "raw_rgb8_bytes":256*256*3,"packet_to_raw_rgb_ratio":budget/(256*256*3),
             "psnr":psnr(original,reconstruction),"mae":mae(original,reconstruction),
             "original_access_at_receiver":received["original_image_access"]}
        for fmt in ["JPEG","WEBP"]:
            control=budget_codec(original,fmt,budget)
            if control:
                name=received["packet"].removesuffix(".zip")+"_"+fmt.lower()+(".jpg" if fmt=="JPEG" else ".webp")
                (output / name).write_bytes(control["payload"])
                row[fmt.lower()]={"bytes":control["bytes"],"quality":control["quality"],
                                  "psnr":psnr(original,control["image"]),"mae":mae(original,control["image"]),
                                  "file":name}
        png=io.BytesIO()
        original.save(png,format="PNG",optimize=True)
        row["png_lossless_bytes"]=len(png.getvalue())
        row["lossless_png_fits_packet_budget"]=len(png.getvalue())<=budget
        thumbnail_file=run / "receiver" / (received["packet"].removesuffix(".zip")+"_thumbnail_only.png")
        if thumbnail_file.exists():
            thumb=Image.open(thumbnail_file).convert("RGB")
            row["thumbnail_only_psnr"]=psnr(original,thumb)
            row["thumbnail_only_mae"]=mae(original,thumb)
        rows.append(row)
    # Feature cosine is a descriptive observation, not an independent semantic metric.
    vectors=[]
    labels=[]
    for record in encoder["rows"]:
        name=f"{record['image']}_s{record['sigma']:.1f}"
        with np.load(run / "features" / (name+".npz")) as archive:
            vectors.append({str(layer):archive[f"layer{layer}"].mean(axis=0).astype(np.float64) for layer in encoder["layers"]})
        labels.append({"image":record["image"],"sigma":record["sigma"]})
    cosines={}
    for layer in map(str,encoder["layers"]):
        matrix=np.stack([entry[layer] for entry in vectors])
        norms=np.linalg.norm(matrix,axis=1,keepdims=True)
        normalized=matrix/np.maximum(norms,1e-12)
        cosine=normalized @ normalized.T
        same,different=[],[]
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):
                if labels[i]["image"]==labels[j]["image"]:
                    same.append(float(cosine[i,j]))
                elif labels[i]["sigma"]==labels[j]["sigma"]:
                    different.append(float(cosine[i,j]))
        cosines[layer]={"matrix":cosine.tolist(),"same_image_different_sigma_mean":statistics.mean(same),
                        "different_image_same_sigma_mean":statistics.mean(different)}
    summaries=[]
    for variant in sorted({row["variant"] for row in rows}):
        subset=[row for row in rows if row["variant"]==variant]
        photographs=[row for row in subset if row["image"] in ["astronaut","coffee","chelsea"]]
        summaries.append({"variant":variant,"n":len(subset),
            "mean_packet_bytes":statistics.mean(row["packet_bytes"] for row in subset),
            "mean_psnr_all":statistics.mean(row["psnr"] for row in subset),
            "mean_psnr_photographs":statistics.mean(row["psnr"] for row in photographs),
            "mean_jpeg_psnr_photographs":statistics.mean(row["jpeg"]["psnr"] for row in photographs),
            "mean_webp_psnr_photographs":statistics.mean(row["webp"]["psnr"] for row in photographs),
            "thumbnail_psnr_gain_photographs":statistics.mean(row["psnr"]-row["thumbnail_only_psnr"] for row in photographs)
                 if "thumbnail_only_psnr" in photographs[0] else None})
    result={"run":args.run,"scope":"3 photographs and 2 synthetic controls; 256px, empty prompt, frozen final head; no adapter training",
            "sender_forwards":len(encoder["rows"]),"captured_feature_tensors":len(encoder["rows"])*len(encoder["layers"]),
            "receiver_packets":len(rows),"comparison_rows":rows,"variant_summary":summaries,
            "feature_cosine":{"labels":labels,"layers":cosines,"scope":"Mean pooled Iris features; not independent semantics evaluation"},
            "oracle_replay":encoder["oracle_replay"],"failures":encoder["failures"],
            "semantic_superiority_tested":False}
    write_json(run / "evaluation-summary.json",result)
    fields=["image","variant","packet_bytes","bpp","packet_to_raw_rgb_ratio","psnr","mae",
            "thumbnail_only_psnr","jpeg_bytes","jpeg_quality","jpeg_psnr","webp_bytes","webp_quality","webp_psnr","png_lossless_bytes"]
    with open(run / "measurements.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat={key:row.get(key) for key in fields}
            for fmt in ["jpeg","webp"]:
                for key in ["bytes","quality","psnr"]:
                    flat[fmt+"_"+key]=row.get(fmt,{}).get(key)
            writer.writerow(flat)
    styles="body{font:15px/1.6 system-ui,sans-serif;margin:32px;background:#f6f8fb;color:#17233a} h1,h2{line-height:1.3} table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px 12px;border-bottom:1px solid #dae1ea;text-align:left} .note{padding:16px;background:#fff4d7;border-left:4px solid #c88a00} .gallery{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:24px}figure{margin:0;background:white;padding:8px}img{width:100%;image-rendering:auto}figcaption{font-size:12px} .section{background:white;padding:18px;margin:20px 0} code{background:#eef2f8;padding:2px 4px}"
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Iris intermediate representation pilot</title><style>'+styles+'</style><body>']
    page+=['<h1>Iris中間表現・通信パケット 初期実験</h1>',
           '<p>学習済みIris-3B / V100 32GB / 256×256 / 写真3枚＋図形2枚。空プロンプト、追加学習なし。</p>',
           '<p class="note"><b>検証範囲：</b>特徴抽出と既存の最終ヘッドを使った通信パケットの動作検証です。独立した意味評価は未実施。元画像を使う単一ステップ推定は診断用であり、通信結果に含めません。受信結果は別プロセスでパケットと共有モデルだけから生成しました。</p>',
           '<h2>パケット容量と忠実性</h2><table><tr><th>方式</th><th>全5枚 平均 bytes</th><th>写真 PSNR</th><th>同予算 JPEG</th><th>同予算 WebP</th><th>サムネイルとの差</th></tr>']
    for summary in summaries:
        delta=summary["thumbnail_psnr_gain_photographs"]
        page.append('<tr><td>'+summary["variant"]+'</td><td>'+f'{summary["mean_packet_bytes"]:,.0f}'+'</td><td>'+f'{summary["mean_psnr_photographs"]:.2f}'+'</td><td>'+f'{summary["mean_jpeg_psnr_photographs"]:.2f}'+'</td><td>'+f'{summary["mean_webp_psnr_photographs"]:.2f}'+'</td><td>'+('—' if delta is None else f'{delta:+.2f} dB')+'</td></tr>')
    page+=['</table><p>PSNRは元画像への忠実性だけを測ります。コーデックはパケット以下の容量で最大品質設定を採用。ZIPヘッダー・特徴・量子化スケール・補助JPEG・seed等のメタデータを含む実ファイル容量です。</p>',
           '<h2>全サンプルの比較</h2>']
    for image_id in sorted({row["image"] for row in rows}):
        page.append('<div class="section"><h3>'+html.escape(image_id)+'</h3>')
        subset=[row for row in rows if row["image"]==image_id]
        chosen=[row for row in subset if row["variant"] in ["full_fp16_none","pool4_int4_none","full_fp16_thumb64","pool4_int4_thumb64","pool2_int4_thumb64"]]
        for row in chosen:
            stem=image_id+"_s0.1_"+row["variant"]
            items=[("../inputs/"+image_id+".png","元画像"),
                   ("../receiver/"+stem+".png",row["variant"]+f' / {row["packet_bytes"]:,} bytes / PSNR {row["psnr"]:.2f}')]
            if "thumbnail_only_psnr" in row:
                items.append(("../receiver/"+stem+"_thumbnail_only.png",f'補助画像のみ / PSNR {row["thumbnail_only_psnr"]:.2f}'))
            items.extend([("evaluation/"+row[fmt]["file"],fmt.upper()+f' / {row[fmt]["bytes"]:,} bytes / PSNR {row[fmt]["psnr"]:.2f}') for fmt in ["jpeg","webp"] if fmt in row])
            # report.html lives at run root, so remove the ../ used for inputs/receiver.
            page.append('<div class="gallery">' + "".join('<figure><img src="'+path.removeprefix("../")+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>' for path,caption in items)+'</div>')
        page.append('</div>')
    page+=['<h2>実験上の限界と次の判断</h2><ul>',
           '<li>256pxは初期メモリ検証のための縮小条件。正式な1MP生成品質の評価ではありません。</li>',
           '<li>最終トークンのプーリングは学習済みボトルネックではありません。失敗しても中間表現全体の可能性を否定できません。</li>',
           '<li>被写体・個数・配置・色の保持は比較画像で観察します。意味的優位性を結論にするには独立評価器と未使用画像が必要です。</li>',
           '<li>通常版には公式の特徴条件デコーダがありません。Adapterや明示的に学習したデコーダの必要性を初期結果から判断します。</li>',
           '</ul><p>元の依頼、事前仮説、環境、チェックポイントと画像のSHA-256、失敗記録はプロジェクトに保存しています。</p></body></html>']
    (run / "report.html").write_text("\n".join(page),encoding="utf-8")
    # Portable contact sheet with real comparison images for direct visual QA.
    photo_ids=["astronaut","coffee","chelsea"]
    cols=["original","pool4_int4_none","pool4_int4_thumb64","thumbnail","jpeg"]
    sheet=Image.new("RGB",(5*256,3*288),"white")
    draw=ImageDraw.Draw(sheet)
    for r,image_id in enumerate(photo_ids):
        row=next(item for item in rows if item["image"]==image_id and item["variant"]=="pool4_int4_thumb64")
        paths=[run/"inputs"/(image_id+".png"),
               run/"receiver"/(image_id+"_s0.1_pool4_int4_none.png"),
               run/"receiver"/(image_id+"_s0.1_pool4_int4_thumb64.png"),
               run/"receiver"/(image_id+"_s0.1_pool4_int4_thumb64_thumbnail_only.png"),
               output/row["jpeg"]["file"]]
        for c,path in enumerate(paths):
            sheet.paste(Image.open(path).convert("RGB"),(c*256,r*288+32))
            draw.text((c*256+6,r*288+7),image_id+" / "+cols[c],fill=(20,30,50))
    sheet.save(run / "contact-sheet.png")
    write_json(run / "status.json",{"phase":"evaluation_complete","receiver_packets":len(rows)})
    print(json.dumps({"run":str(run),"variant_summary":summaries,"failures":encoder["failures"]},indent=2),flush=True)

if __name__=="__main__":
    main()
