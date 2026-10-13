"""Validate the initial evidence ledger and preserve factual Japanese research notes."""
import hashlib
import json
import statistics
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/"results/pilot-20261013"
enc=json.loads((run/"encoder-summary.json").read_text())
rx=json.loads((run/"receiver-summary.json").read_text())
ev=json.loads((run/"evaluation-summary.json").read_text())
assert len(enc["rows"])==15 and len(rx["rows"])==30
assert sum(len(row["layers"]) for row in enc["rows"])==75
assert all(info["finite"] for row in enc["rows"] for info in row["layers"].values())
assert all(row["max_velocity_error"]==0 for row in enc["oracle_replay"])
assert not enc["failures"]
assert rx["original_image_access"] is False
assert all(row["original_image_access"] is False for row in rx["rows"])
manifest=json.loads((ROOT/"data/manifest.json").read_text())
for sample in manifest["images"]:
    assert hashlib.sha256((ROOT/sample["file"]).read_bytes()).hexdigest()==sample["sha256"]
for row in ev["comparison_rows"]:
    for codec in ["jpeg","webp"]:
        assert row[codec]["bytes"]<=row["packet_bytes"]
class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets=[]
    def handle_starttag(self,tag,attrs):
        if tag=="img":
            self.assets.extend(v for k,v in attrs if k=="src")
parser=Assets()
parser.feed((run/"report.html").read_text(encoding="utf-8"))
missing=[path for path in parser.assets if not (run/path).is_file()]
assert not missing,missing
photo_rows=[row for row in ev["comparison_rows"] if row["image"] in ["astronaut","coffee","chelsea"]]
variants=["full_fp16_none","full_int8_none","pool4_int4_none","full_fp16_thumb64","pool4_int4_thumb64","pool2_int4_thumb64"]
lines=[]
for variant in variants:
    rows=[row for row in photo_rows if row["variant"]==variant]
    size=statistics.mean(row["packet_bytes"] for row in rows)
    value=statistics.mean(row["psnr"] for row in rows)
    jpeg=statistics.mean(row["jpeg"]["psnr"] for row in rows)
    webp=statistics.mean(row["webp"]["psnr"] for row in rows)
    lines.append(f"| {variant} | {size:,.0f} | {value:.2f} | {jpeg:.2f} | {webp:.2f} |")
peak=max(row["telemetry"]["torch_peak_allocated_bytes"] for row in enc["rows"])/2**30
gpu_temp=max(row["telemetry"]["cooling"]["gpu"]["Temperature"] for row in enc["rows"])
hbm_temp=max(row["telemetry"]["cooling"]["gpu"]["MemoryTemperature"] for row in enc["rows"])
warm=statistics.median(row["forward_seconds"] for row in enc["rows"][1:])
note=f"""# 初期実験の結果 / 2026-10-13

独立プロジェクトで学習済みIris-3BをV100上で実行し、特徴抽出と、元画像を読まない別プロセスによるパケット復号を検証した。今回の既存最終ヘッドをそのまま流用する方式では、圧縮の利点を確認できなかった。これはIrisの中間表現一般やセマンティック通信全体の否定ではなく、特定の接続方式・小規模条件での結果。

## 検証した条件

- 通常版Iris、コードrevision cee0c130a62a9c288ad8918f1f7cdf23cc228d69、重みrevision adc05b909fb05d4276d26aae5a52b36406b2ef26。
- 写真3枚と自作図形2枚、RGB 256×256、空プロンプト。入力出典・ライセンス・加工・SHA-256はdata/manifest.json。
- ノイズ量0.1/0.5/0.9、全画像で同一CPU乱数seed 20261013。画像差と乱数差を混ぜない条件。
- 1/8/10/16/24層の特徴を計75テンソル抽出。全て有限。
- パケット復号はsigma=0.1、最終24層、凍結したPiTヘッド、FP32パラメータ＋FP16 autocast、math SDPA。追加学習はゼロ。
- 受信は別コマンド・別プロセス。共有モデルとパケットのみを読み、元画像は評価段階まで読まない。
- 特徴だけの場合は公開ノイズとゼロのクリーン画像を仮定。補助画像ありの場合は転送した64×64 JPEGからnoisy RGBを近似する。
- 元のnoisy RGBを与えるoracle再実行では、元モデルとの最大velocity誤差は5枚全て0.0。oracleは通信結果から除外した。

## 容量と忠実性（以下は写真3枚だけの平均）

| 方式 | 実パケット bytes | 復号PSNR dB | 同予算以下JPEG | 同予算以下WebP |
|---|---:|---:|---:|---:|
"""+"\n".join(lines)+f"""

ZIPヘッダー、JSONメタデータ、量子化スケール、補助JPEG、seed等を全て含む。JPEG/WebPはパケット以下の実サイズになる最大品質設定を採用した。モデルと固定空プロンプトは双方にある共有資産として通信量から除外し、そのサイズをmodel-manifestに別記した。

特徴1層は256×2560成分。FP16の非圧縮値は1,310,720 bytesで、RGB8の196,608 bytesの6.67倍。特徴をそのまま取り出すだけでは圧縮にならない。

64×64補助JPEGの単純な拡大だけでは写真平均24.82dB。full_fp16_thumb64は約0.88dB、pool4_int4_thumb64は約1.60dB、pool2_int4_thumb64は約1.78dB低下した。今回の凍結ヘッドでは、特徴を追加することによる忠実性改善を確認できない。

## 画像で観察した内容

特徴のみの出力はノイズと弱い輪郭・色変化が主体で、元写真を十分に復元しない。補助画像がある出力には被写体の大まかな配置・色が戻るが、細部はぼけ、ノイズが残る。全例をreport.htmlに保存した。成功に見える例だけを選んでいない。

## 実行確認

- 推論15回、特徴75個、送信パケット30個、独立受信30個。非有限値と実行失敗は0。
- モデル1forwardのwarm中央値 {warm:.4f}秒。ロード、CPU保存、量子化、画像処理を含む総時間ではない。
- PyTorchのpeak allocated memory {peak:.2f}GiB。CUDAコンテキスト等を含むGPU総使用量とは区別する。
- 記録時点の最高温度はGPU {gpu_temp}℃、HBM {hbm_temp}℃。短い初期試験であり、長時間負荷の冷却検証ではない。
- 量子化の誤差上限・奇数長int4・パケットサイズ・非有限値拒否のテスト5件が通った。
- 画像SHA-256、30件の受信記録、60個のコーデック結果、レポート内{len(parser.assets)}個の画像参照を確認した。
- 通常の一括取得は進まなかったため、32MiBの分割取得へ変更。全11,950,112,680 bytesを結合し、公開LFS SHA-256 a90e1f4c197bac8491f82078def68884b6a3527ea7fc97a693e3e367372aa80f と一致した。転送・検査の経過はresults/download-progress.json。

## 考察と未検証事項

既存PiTのnoisy RGB入力とflowのx-sigma*vという経路への依存が、特徴だけの復元を難しくしていると考えられる。入力分布の変更、sigma、特徴加工それぞれの寄与はまだ分離していない。

PSNRは忠実性だけを評価する。被写体・数・配置・色等を独立した評価器と人手で判定する意味評価は未実施。5枚、1seed、256px、単一ステップ、空プロンプトという限定条件であり、正式な1024px生成、テキスト条件、全生成過程、他の潜在モデル、Adapter学習、汎化性能は未検証。

E001の層別cosine行列はevaluation-summary.jsonに保存した。Iris自身の特徴を用いた記述的観察であり、独立した意味評価とは扱わない。

次段階は、低容量特徴から生成へつなぐ復号器・Adapterと、未使用画像による独立した意味評価の設計を優先する。詳しい判断はnext-experiments.md。
"""
(ROOT/"docs/initial-results-ja.md").write_text(note,encoding="utf-8")
log=ROOT/"docs/research-log.md"
if "## Initial milestone completed" not in log.read_text(encoding="utf-8"):
    with log.open("a",encoding="utf-8") as f:
        f.write("\n## Initial milestone completed\n\nPretrained V100 execution: 15 forwards, 75 finite feature tensors, 30 independent receiver packets. Oracle head replay matched exactly for all 5 inputs. The frozen-head packet pilot did not improve fidelity over its thumbnail-only control and trailed byte-budget-matched JPEG/WebP. See initial-results-ja.md and results/pilot-20261013/report.html. Semantic superiority and a learned feature-conditioned decoder remain unverified.\n")
completion={"initial_milestone":"complete","whole_research_project":"ongoing",
            "confirmed":["trained V100 inference","feature extraction","actual packet byte accounting","independent receiver",
                         "frozen head oracle parity","byte-budget codec comparison","local visual artifacts"],
            "unverified":["semantic advantage","learned adapter","held-out generalization","other generators","full generation reconstruction"],
            "validation":{"feature_tensors":75,"packets":30,"report_image_refs":len(parser.assets),"missing_image_refs":0,
                          "oracle_max_error":0.0,"nonfinite":0,"unit_tests_passed":5}}
(run/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(completion,indent=2),flush=True)
