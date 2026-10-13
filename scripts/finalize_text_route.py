"""Audit actual completion and produce a Japanese evidence-backed research note."""
import re
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_route_common import ROOT, RUN, PROTOCOL, protocol, cases, all_captions, read, save, sha, now
from build_text_route_report import NAMES

def main():
    plan = protocol()
    captions = all_captions()
    gen = read(RUN / "generation.json")
    refined = read(RUN / "generation-refine40.json")
    encoding = read(RUN / "text-encoding.json")
    if len(captions) != 78 or any(r["status"] != "completed" for r in captions):
        raise ValueError("Caption coverage incomplete")
    if len(gen) != 158 or any(r["status"] != "completed" for r in gen):
        raise ValueError("Main reconstruction coverage incomplete")
    if len(refined) != 18 or any(r["status"] != "completed" for r in refined):
        raise ValueError("Refinement coverage incomplete")
    expected = {(c["id"], n, b, seed) for c in cases() for n in plan["models"] for b in plan["budgets"] for seed in plan["seeds"]}
    actual = {(r["case_id"], r["model"], r["budget"], r["seed"]) for r in gen if r["model"] != "empty_control"}
    if actual != expected:
        raise ValueError("Condition identities do not match full comparison grid")
    checked = 0
    for r in gen + refined:
        if sha(RUN / r["file"]) != r["sha256"] or sha(RUN / r["native_file"]) != r["native_sha256"]:
            raise ValueError("Output integrity failed")
        if (RUN / r["packet_file"]).stat().st_size != r["wire_bytes"] or sha(RUN / r["packet_file"]) != r["packet_sha256"]:
            raise ValueError("Complete packet byte accounting failed")
        packet = read(RUN / r["packet_file"])
        if set(packet) != {"protocol", "prompt", "seed", "width", "height"} or packet["prompt"] != r["prompt"]:
            raise ValueError("Non-text input or changed prompt in wire packet")
        checked += 1
    if len(encoding) != 79 or any(r["status"] != "completed" or r["caption_tokens"] > 295 for r in encoding):
        raise ValueError("Text encoding failures or hidden prompt overflow")
    native_sender = read(RUN / "native-feature-check/sender.json")
    native_receiver = read(RUN / "native-feature-check/receiver.json")
    if len(native_sender["rows"]) != 9 or any(r["oracle_velocity_max_error"] != 0 for r in native_sender["rows"]):
        raise ValueError("Native head extraction/replay failed")
    if len(native_receiver["rows"]) != 18 or native_receiver["source_image_access"]:
        raise ValueError("Native independent receiver coverage failed")
    summary = read(RUN / "evaluation-summary.json")
    metrics = read(RUN / "metrics.json")
    text = [r for r in summary["photo_aggregates"] if r["route"] == "text_only"]
    if len(text) != 6 or any(r["photos"] != 11 or r["generated_observations"] != 22 for r in text):
        raise ValueError("Photo-only aggregate/seed grouping failed")
    if sum(r["route"] == "text_only" for r in metrics) != 156 or sum(r["route"] == "text_refine40" for r in metrics) != 18:
        raise ValueError("Evaluator output coverage incomplete")
    if any(r["paired_photos"] != 3 for r in summary["refinement_same_seed"]):
        raise ValueError("Refinement pairing failed")
    report = (RUN / "report.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:href|src)="([^"#]+)"', report)
    missing = [s for s in references if not s.startswith(("http://", "https://")) and not (RUN / s).exists()]
    if missing:
        raise ValueError("Broken report references: " + str(missing))
    by_group = {(r["model"], r["budget"]): r for r in text}
    lines = ["# 画像→テキスト→画像の比較結果", "", "独立プロジェクトで未確認だった経路を実際に検証しました。Qwen3-VL-4B-Instruct、MiniCPM-V4.6、LFM2.5-VL-1.6Bの3候補を、同じ入力・生成器・乱数種で比較しています。", "",
             "標準候補はQwen3-VL-4B-Instructとします。短文制約を全13入力で守り、Irisの必須テキスト符号化器も兼ねるためです。品質の平均値と通信量は下表で個別に判断し、総合的・一般的な優位を確定したとは扱いません。", "",
             "| モデル | 説明 | 平均通信量 B | CLIP画像類似度 | DINO画像類似度 | PSNR dB | 語数制約内 /13 | VLM中央値 秒 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in plan["models"]:
        for budget in plan["budgets"]:
            row = by_group[(name, budget)]
            cap = [r for r in captions if r["model"] == name and r["budget"] == budget]
            lines.append(f"| {NAMES[name]} | {budget} | {row['wire_bytes']:.0f} | {row['clip_image_cosine']:.3f} | {row['dino_image_cosine']:.3f} | {row['psnr_db']:.2f} | {sum(r['within_word_target'] for r in cap)}/13 | {statistics.median(r['inference_seconds'] for r in cap):.2f} |")
    codec = next(r for r in summary["same_text_ceiling_codec_aggregates"] if (r["sender_model"], r["caption_budget"], r["format"], r["mode"]) == ("qwen4b", "detailed", "WEBP", "adaptive_resolution"))
    lines += ["", f"同じQwen詳細パケットのバイト上限内で送った縮小WebPは、写真{codec['photos_encoded']}枚・平均{codec['wire_bytes']:.0f}B、CLIP {codec['clip_image_cosine']:.3f}、DINO {codec['dino_image_cosine']:.3f}、PSNR {codec['psnr_db']:.2f}dBでした。文章再生成は意味類似度で有利、縮小WebPは画素一致で有利です。固定256pxのままでは小さい上限に収まらないJPEG／WebP条件もあり、その失敗も記録しています。"]
    lines += ["", "通信量と画像指標は写真11枚、各写真内で2シードを平均した値です。語数・VLM時間は図形を含む13入力。モデルの取得・配布費用は共有済みモデルを前提として別扱いです。文章だけで場面は描けますが、同じ顔・個体・形・構図の復元を保証できません。", "",
              "## 比較の条件と限界", "", "入力13枚（写真11＋図形2）、3候補、短文40語／詳細100語の依頼、2シードで156生成。空文2件を加え158件。Irisは1024px・DPM++・10ステップ・CFG3。元の3写真は同じ第一シードで40ステップも18件生成しました。評価は元入力と同じ256pxへ縮小。正式推奨の100ステップ比較ではありません。", "",
              "最初の256px生成はFP16／FP32とも縞模様、512pxでも崩れが出たため、比較から除外して記録を残しました。標準1024pxで有効な画像生成を確認して全条件を揃えています。VLMの成否と解像度に起因する生成失敗を混同しません。", "",
              "受信側は文章・乱数種・共通プロトコル・共有モデルだけを使用。画像に由来するVLMの隠れ状態、原画像、補助画像を渡していません。文章から受信側で作った状態のキャッシュは通信量に含める送信情報ではありません。全78説明が295トークン以内に収まり、説明の切り捨てはありません。", "",
              "CLIP／DINOの高さは意味・外観の参考値です。DINO系はIrisのREPA設計と重なります。左右・個数・本人性を保証する指標ではありません。PSNRは画素一致。95%区間は11写真を単位とする記述的ブートストラップで、多重比較補正はしていません。既知画像が含まれ、モデルの学習データとの重複も不明です。", "",
              "## 中間特徴との追加比較", "", "初回の特徴検証が256pxだったため、同じ元の3写真を1024pxへ拡大して特徴抽出・独立復元を再確認しました。最終層、σ=0.1、完全FP16／4×4 pooled int4、64px JPEG有無、特徴ゼロ・位置シャッフルを比較。実際のZIP全体、量子化スケール、JPEG、メタデータを数えています。これは凍結ヘッドの診断で、学習済みの特徴コーデックではありません。", "",
              "| 方法 | 平均通信 B | CLIP | DINO | PSNR dB |", "|---|---:|---:|---:|---:|"]
    native = [r for r in summary["photo_aggregates"] if r["route"] in ["native_feature_head", "native_thumbnail"]]
    for r in native:
        lines.append(f"| {r['budget']} | {r['wire_bytes']:.0f} | {r['clip_image_cosine']:.3f} | {r['dino_image_cosine']:.3f} | {r['psnr_db']:.2f} |")
    lines += ["", "特徴ゼロ／シャッフルとの違いから特徴が情報を持つことは確認できます。一方、特徴＋補助画像は補助画像単体よりPSNR／DINOが低く、現行方式の通信効率上の優位を示せていません。文章方式との通信量・計算量・精度設定は異なるため、中間表現一般の優劣は結論しません。", "",
              "## 読み取りと再生成で見えた問題", "", "Qwenの短文でも『framed photograph』から原画像にない額縁が生成されました。青い四角はQwenの詳細文・MiniCPMで中央と説明され、LFMは赤丸を中央と説明しました。Qwen／LFMは鳥の向きを左右逆に説明する例があり、LFMにはカップの満杯状態や取っ手の方向の誤りもありました。詳細はcaption-factual-spot-checks.jsonに残しています。これは独立した人間評価の総合得点ではありません。", "",
              "次の改善候補は、読み取りの左右・上下・個数・クロップを明示する説明契約と、文章で落ちる構造／個体情報だけを追加する小さな補助表現です。現行の大きな特徴をそのまま送る方法を続けるより、強い文章ベースラインを維持し、同じ通信量でどの情報が追加されるか検証する必要があります。学習済みアダプタや画像条件付き生成は今回は未実施です。", "",
              "## 成果物", "", "- [画像比較レポート](../results/text-route-20261013/report.html)", "- [集計・差の区間](../results/text-route-20261013/evaluation-summary.json)", "- [測定CSV](../results/text-route-20261013/metrics.csv)", "- [モデルと通信量の図](../results/text-route-20261013/model-comparison.png)", "- [3写真の通信量と品質](../results/text-route-20261013/communication-tradeoff.png)", "- [40ステップの比較](../results/text-route-20261013/refinement-compact.png)", "- [再現手順](text-route-reproduction.md)", "- [条件・診断記録](text-route-protocol.md)", "",
              "一次資料： [Qwenモデルカード](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、[MiniCPMモデルカード](https://huggingface.co/openbmb/MiniCPM-V-4.6)、[LFMモデルカード](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)、[Iris公式](https://huggingface.co/speridlabs/iris-3b)。各原写真の作者・ライセンスはdataset.jsonと画像レポートに記録しています。", ""]
    refinement_lines = ["## 40ステップの追加確認", "", "同じ説明・元写真・第一シードを使った40−10ステップの差です。対象は元の3写真だけです。", "",
                        "| モデル | 説明 | CLIP差 | DINO差 | PSNR差 dB |", "|---|---|---:|---:|---:|"]
    for r in summary["refinement_same_seed"]:
        delta = r["metrics"]
        refinement_lines.append(f"| {NAMES[r['model']]} | {r['budget']} | {delta['clip_image_cosine']['mean_difference']:+.3f} | {delta['dino_image_cosine']['mean_difference']:+.3f} | {delta['psnr_db']['mean_difference']:+.2f} |")
    refinement_lines += ["", "この3写真では、ステップ数の増加で元の画素へ一様に近づく効果は見られませんでした。意味類似指標の変化もモデル・説明によって異なります。生成を細かくすることだけで、説明に欠ける個体情報・構造が戻るとは言えません。正式100ステップの検証は未実施です。", ""]
    position = lines.index("## 読み取りと再生成で見えた問題")
    lines[position:position] = refinement_lines
    (ROOT / "docs/text-route-results-ja.md").write_text("\n".join(lines), encoding="utf-8")
    completion = {"at": now(), "status": "complete", "caption_records": len(captions), "main_generated_images": 156, "empty_controls": 2,
                  "refinement_images": len(refined), "native_feature_packets": 9, "native_feature_decoded_controls": 18, "evaluated_records": len(metrics),
                  "verified_output_and_packet_pairs": checked, "report_local_references_checked": len(references) - sum(s.startswith(("http://", "https://")) for s in references),
                  "protocol_sha256": sha(PROTOCOL), "limitations": plan["limitations"]}
    save(RUN / "completion.json", completion)
    save(RUN / "status.json", completion)
    print("TEXT_ROUTE_COMPLETE", completion, flush=True)

if __name__ == "__main__":
    main()
