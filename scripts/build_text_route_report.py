"""Inspectable comparison sheets, scientific plots and local HTML report."""
import argparse
import csv
import html
import statistics
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".plot-deps"))
from PIL import Image, ImageDraw
import numpy as np
from text_route_common import RUN, protocol, cases, all_captions, read, save, now

NAMES = {"qwen4b": "Qwen3-VL-4B", "minicpm": "MiniCPM-V4.6", "lfm16": "LFM2.5-VL-1.6B"}
COLORS = {"qwen4b": "#2463b5", "minicpm": "#b34b40", "lfm16": "#16806b"}

def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None

def sheets(generation):
    lookup = {(r["case_id"], r["model"], r["budget"], r["seed"]): r for r in generation if r["status"] == "completed"}
    selected = cases()
    cell, row_height = 192, 230
    headers = ["Original"] + [NAMES[n] + " " + b for n in protocol()["models"] for b in ["compact", "detailed"]]
    for seed in protocol()["seeds"]:
        for start in range(0, len(selected), 5):
            group = selected[start:start + 5]
            canvas = Image.new("RGB", (cell * 7, 40 + row_height * len(group)), "#f4f5f7")
            draw = ImageDraw.Draw(canvas)
            for col, title in enumerate(headers):
                draw.text((col * cell + 5, 10), title, fill="black")
            for i, case in enumerate(group):
                y = 40 + i * row_height
                with Image.open(ROOT / case["file"]) as im:
                    canvas.paste(im.convert("RGB").resize((cell, cell)), (0, y))
                draw.text((5, y + cell + 4), case["id"], fill="black")
                for col, (name, budget) in enumerate([(n, b) for n in protocol()["models"] for b in ["compact", "detailed"]], 1):
                    row = lookup.get((case["id"], name, budget, seed))
                    if row:
                        with Image.open(RUN / row["file"]) as im:
                            canvas.paste(im.convert("RGB").resize((cell, cell)), (col * cell, y))
                        draw.text((col * cell + 5, y + cell + 4), str(row["wire_bytes"]) + " bytes", fill="black")
            canvas.save(RUN / f"comparison-s{seed}-part{start // 5 + 1}.png")
    native = RUN / "native-feature-check/receiver.json"
    if native.exists():
        feature = {(r["case_id"], r["variant"]): r for r in read(native)["rows"]}
        cols = ["Original", "Feature only", "Feature + thumb", "Thumb only", "Zero feat + thumb", "Shuffled + thumb", "Qwen text"]
        canvas = Image.new("RGB", (7 * 224, 3 * 256 + 35), "white")
        draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(cols):
            draw.text((j * 224 + 5, 8), label, fill="black")
        for i, case in enumerate(selected[:3]):
            paths = [ROOT / case["file"]]
            paths += [RUN / feature[(case["id"], variant)]["file"] for variant in ["pool4_int4_none", "pool4_int4_thumb64", "thumbnail64", "pool4_int4_thumb64_zero", "pool4_int4_thumb64_shuffle"]]
            qwen = lookup.get((case["id"], "qwen4b", "compact", protocol()["seeds"][0]))
            paths += [RUN / qwen["file"] if qwen else None]
            for j, path in enumerate(paths):
                if path:
                    with Image.open(path) as im:
                        canvas.paste(im.convert("RGB").resize((224, 224)), (j * 224, 35 + i * 256))
            draw.text((6, 35 + i * 256 + 228), case["id"], fill="black")
        canvas.save(RUN / "native-feature-comparison.png")
    refined_path = RUN / "generation-refine40.json"
    if refined_path.exists():
        refined = {(r["case_id"], r["model"], r["budget"]): r for r in read(refined_path) if r["status"] == "completed"}
        seed = protocol()["refinement_check"]["seed"]
        for budget in ["compact", "detailed"]:
            canvas = Image.new("RGB", (7 * 224, 3 * 256 + 35), "white")
            draw = ImageDraw.Draw(canvas)
            labels = ["Original"] + [NAMES[n] + " " + str(step) + " steps" for n in protocol()["models"] for step in [10, 40]]
            for j, label in enumerate(labels):
                draw.text((j * 224 + 4, 8), label, fill="black")
            for i, case in enumerate(selected[:3]):
                with Image.open(ROOT / case["file"]) as im:
                    canvas.paste(im.convert("RGB").resize((224, 224)), (0, 35 + i * 256))
                for j, (name, step) in enumerate([(n, s) for n in protocol()["models"] for s in [10, 40]], 1):
                    row = lookup.get((case["id"], name, budget, seed)) if step == 10 else refined.get((case["id"], name, budget))
                    if row:
                        with Image.open(RUN / row["file"]) as im:
                            canvas.paste(im.convert("RGB").resize((224, 224)), (j * 224, 35 + i * 256))
                draw.text((5, 35 + i * 256 + 228), case["id"], fill="black")
            canvas.save(RUN / ("refinement-" + budget + ".png"))

def plots(metrics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aggregate = {(r["model"], r["budget"]): r for r in summary["photo_aggregates"] if r["route"] == "text_only"}
    arms = [(n, b) for n in protocol()["models"] for b in ["compact", "detailed"]]
    if all(arm in aggregate for arm in arms):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        labels = [NAMES[n] + (" / compact" if b == "compact" else " / detailed") for n, b in arms]
        for ax, key, title in zip(axs, ["clip_image_cosine", "dino_image_cosine"], ["CLIP image cosine", "DINOv2 image cosine"]):
            for i, arm in enumerate(arms):
                r = aggregate[arm]
                value = r[key]
                low, high = r[key + "_image_bootstrap95"]
                ax.errorbar(value, i, xerr=[[value - low], [high - value]], fmt="o", color=COLORS[arm[0]], capsize=4)
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xlim(0.35, 1.0)
            ax.grid(axis="x", alpha=.25)
            ax.set_title(title)
        fig.suptitle("11 photos, means over 2 seeds; image bootstrap 95% intervals")
        fig.savefig(RUN / "model-comparison.png", dpi=160)
        fig.savefig(RUN / "model-comparison.pdf")
        plt.close(fig)
    common = [r for r in metrics if r["case_id"] in ["astronaut", "coffee", "chelsea"]]
    points = []
    selections = [("native_feature_head", "Iris_frozen_head", b, "Feature " + b) for b in ["full_fp16_none", "pool4_int4_none", "pool4_int4_thumb64"]]
    selections += [("native_thumbnail", "thumbnail64", "thumbnail64", "Thumb64")]
    selections += [("text_only", n, b, NAMES[n] + " / " + b) for n in protocol()["models"] for b in ["compact", "detailed"]]
    selections += [("initial_codec", "JPEG", "pool4_int4_thumb64", "JPEG ~30k ceiling")]
    for route, name, budget, label in selections:
        rows = [r for r in common if (r["route"], r["model"], r["budget"]) == (route, name, budget)]
        if not rows:
            continue
        points.append({"route": route, "model": name, "budget": budget, "label": label, "source_photos": len(set(r["case_id"] for r in rows)),
                       **{k: mean(rows, k) for k in ["wire_bytes", "clip_image_cosine", "dino_image_cosine", "psnr_db"]}})
    for budget in ["compact", "detailed"]:
        rows = [r for r in common if r["route"] == "codec" and r["sender_model"] == "qwen4b" and r["caption_budget"] == budget and r["model"] == "WEBP" and r["budget"] == "adaptive_resolution"]
        if rows:
            points.append({"route": "codec", "model": "WEBP", "budget": budget, "label": "Scaled WebP / Qwen " + budget + " ceiling", "source_photos": len(rows),
                           **{k: mean(rows, k) for k in ["wire_bytes", "clip_image_cosine", "dino_image_cosine", "psnr_db"]}})
    save(RUN / "matched-three-photo-points.json", points)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for p in points:
        color = COLORS.get(p["model"], "#946eaf" if p["model"] == "WEBP" else "#555555")
        marker = "o" if p["route"] == "text_only" else "^" if p["route"] == "native_thumbnail" else "s"
        for ax, key in zip(axs, ["clip_image_cosine", "psnr_db"]):
            ax.scatter(p["wire_bytes"], p[key], color=color, marker=marker, label=p["label"], s=45)
            ax.set_xscale("log")
            ax.grid(alpha=.2)
            ax.set_xlabel("Complete per-image transmitted bytes (log scale)")
    axs[0].set_ylabel("CLIP image cosine")
    axs[1].set_ylabel("PSNR at common256 reference (dB)")
    axs[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Same 3 photos; different codecs/compute, pretrained assets shared")
    fig.savefig(RUN / "communication-tradeoff.png", dpi=160)
    fig.savefig(RUN / "communication-tradeoff.pdf")
    plt.close(fig)

def html_report(generation, metrics, summary, partial):
    captions = {(r["case_id"], r["model"], r["budget"]): r for r in all_captions()}
    generation_lookup = {(r["case_id"], r["model"], r["budget"], r["seed"]): r for r in generation if r["status"] == "completed" and r["model"] != "empty_control"}
    esc = html.escape
    body = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>画像→文章→画像 比較</title>',
            '<style>body{font-family:Segoe UI,Meiryo,sans-serif;background:#f6f7f9;color:#172534;margin:24px}h1{font-size:27px}p{line-height:1.7;max-width:1000px}.grid{display:grid;grid-template-columns:repeat(7,minmax(135px,1fr));gap:10px}section{background:white;padding:18px;margin:22px 0;border-radius:12px}figure{margin:0}img{width:100%;height:auto}figcaption{font-size:12px;line-height:1.5}details{font-size:12px;line-height:1.5;margin-top:7px}table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.note{color:#596773}.controls{position:sticky;top:0;background:#fff;padding:12px;border-bottom:1px solid #ddd}select{padding:6px;margin-right:15px}.seed{display:none}.seed.active{display:block}@media(max-width:1000px){.grid{grid-template-columns:repeat(3,minmax(135px,1fr))}}</style>',
            '<h1>画像→テキスト→画像：3モデル比較</h1>',
            f'<p>{"実行途中の確認用" if partial else "比較完了"}。原画像13枚（写真11＋図形2）、3モデル、短文・詳細文、2シード。保存済み生成：{len(generation)}件。Irisは標準1024px／10ステップ。評価用画像は256pxへ縮小。元の3写真は同じシードで40ステップも確認します。</p>',
            '<p class="note">共有済みのモデルを前提とした通信量です。CLIP／DINOは意味・見た目の類似度の参考値で、本人の顔・個体・形状の一致を保証しません。PSNRは元の画素との一致です。小規模な試験で、学習データとの重複は不明です。</p>']
    if summary:
        body.append('<table><tr><th>モデル／説明</th><th>平均通信B</th><th>CLIP</th><th>DINO</th><th>PSNR dB</th></tr>')
        for row in summary["photo_aggregates"]:
            if row["route"] != "text_only":
                continue
            body.append(f'<tr><td>{NAMES[row["model"]]}／{row["budget"]}</td><td>{row["wire_bytes"]:.0f}</td><td>{row["clip_image_cosine"]:.3f}</td><td>{row["dino_image_cosine"]:.3f}</td><td>{row["psnr_db"]:.2f}</td></tr>')
        body.append('</table><p>写真11枚のみ。各写真内で2シードを平均しています。指標は個別に読む必要があります。</p>')
        codec = next((r for r in summary.get("same_text_ceiling_codec_aggregates", []) if (r["sender_model"], r["caption_budget"], r["format"], r["mode"]) == ("qwen4b", "detailed", "WEBP", "adaptive_resolution")), None)
        if codec:
            body.append(f'<p>Qwen詳細と同じバイト上限内の縮小WebP（写真{codec["photos_encoded"]}枚）は平均{codec["wire_bytes"]:.0f}B、CLIP {codec["clip_image_cosine"]:.3f}、DINO {codec["dino_image_cosine"]:.3f}、PSNR {codec["psnr_db"]:.2f}dBでした。文章再生成は意味類似度が高く、縮小画像は画素一致が高いという違いがあります。</p>')
        body.append('<p><a href="model-comparison.png">モデルの比較図</a> ／ <a href="communication-tradeoff.png">同じ3写真での通信量と品質</a> ／ <a href="native-feature-comparison.png">中間特徴・補助画像との比較</a> ／ <a href="evaluation-summary.json">集計と差の区間</a></p>')
        body.append('<p>同じ説明・入力・シードでの10→40ステップ比較：<a href="refinement-compact.png">短文</a> ／ <a href="refinement-detailed.png">詳細文</a>。対象は元の3写真です。</p>')
    body.append('<div class="controls">シード <select id="seed"><option>20261013</option><option>20261014</option></select>モデル <select id="model"><option value="all">すべて</option>' + ''.join(f'<option value="{n}">{title}</option>' for n, title in NAMES.items()) + '</select>説明 <select id="budget"><option value="all">短文＋詳細</option><option value="compact">短文</option><option value="detailed">詳細</option></select></div>')
    for case in cases():
        image_ref = "../../" + case["file"].replace("\\", "/")
        body.append(f'<section><h2>{esc(case["id"])} <small>{esc(case["category"])}</small></h2><div class="grid"><figure><a href="{esc(image_ref)}"><img src="{esc(image_ref)}"></a><figcaption>入力：全モデルで同じ256px</figcaption></figure>')
        for name in protocol()["models"]:
            for budget in ["compact", "detailed"]:
                caption = captions[(case["id"], name, budget)]
                body.append(f'<figure data-model="{name}" data-budget="{budget}"><figcaption><b>{NAMES[name]}／{budget}</b></figcaption>')
                for seed in protocol()["seeds"]:
                    row = generation_lookup.get((case["id"], name, budget, seed))
                    body.append(f'<div class="seed {"active" if seed == protocol()["seeds"][0] else ""}" data-seed="{seed}">')
                    if row:
                        body.append(f'<a href="{esc(row["native_file"])}"><img loading="lazy" src="{esc(row["file"])}"></a><figcaption>{row["wire_bytes"]}B／{seed}／画像をクリックで1024px</figcaption>')
                    else:
                        body.append('<p>生成待ち</p>')
                    body.append('</div>')
                body.append(f'<details><summary>実際の説明（{caption.get("words", 0)}語）</summary>{esc(caption.get("prompt", caption.get("error", "")))}</details></figure>')
        body.append('</div>')
        if case.get("source_page"):
            body.append(f'<p class="note">原写真：{esc(case["author"])}／{esc(case["title"])}。<a href="{esc(case["source_page"])}">出典</a>。<a href="{esc(case["image_license"])}">CC BY 2.0</a>。この研究では中央クロップ・縮小を行いました。</p>')
        else:
            body.append(f'<p class="note">原画像：{esc(case.get("credit", "Project synthetic control"))}／{esc(case.get("license", ""))}</p>')
        body.append('</section>')
    body.append('<p><a href="../../docs/text-route-protocol.md">条件・失敗診断</a> ／ <a href="metrics.csv">全測定CSV</a> ／ <a href="captions/qwen4b.json">Qwen出力</a> ／ <a href="captions/minicpm.json">MiniCPM出力</a> ／ <a href="captions/lfm16.json">LFM出力</a></p><script>function update(){let s=document.getElementById("seed").value,m=document.getElementById("model").value,b=document.getElementById("budget").value;document.querySelectorAll("[data-model]").forEach(x=>x.style.display=(m==="all"||x.dataset.model===m)&&(b==="all"||x.dataset.budget===b)?"":"none");document.querySelectorAll(".seed").forEach(x=>x.classList.toggle("active",x.dataset.seed===s))}document.querySelectorAll("select").forEach(x=>x.addEventListener("change",update));update()</script></html>')
    (RUN / ("report-progress.html" if partial else "report.html")).write_text('\n'.join(body), encoding="utf-8")

def main(args):
    generation = read(RUN / "generation.json")
    metrics = read(RUN / "metrics.json") if (RUN / "metrics.json").exists() else []
    summary = read(RUN / "evaluation-summary.json") if (RUN / "evaluation-summary.json").exists() and not args.partial else None
    sheets(generation)
    if summary:
        plots(metrics, summary)
    keys = ["route", "model", "budget", "case_id", "kind", "seed", "wire_bytes", "clip_image_cosine", "dino_image_cosine", "psnr_db", "mae_rgb8", "file"]
    with open(RUN / "metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    html_report(generation, metrics, summary, args.partial)
    print("REPORT_BUILT", "partial" if args.partial else "final", len(generation), len(metrics), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true")
    main(p.parse_args())
