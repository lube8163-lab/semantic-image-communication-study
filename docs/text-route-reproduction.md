# 公開版の再現・検証手順

## 公開結果を検証する（GPU・モデル不要）

Python 3.12で `python scripts/verify_publication.py` を実行します。13入力、176生成画像と対応する実テキストパケット、固定プロトコルのSHA-256を照合し、459測定行から写真単位の6条件×5指標の平均を再集計します。これは保存結果の整合性確認であり、モデルの推論再現ではありません。

`results/text-route-20261013/report.html` はローカルで開けます。記事の静的サイトは `requirements-site.txt` をインストールして `python scripts/build_site.py`、`python -m http.server 8765 --directory site` で表示できます。

## 新しい推論実験用の環境

実測はWindows/Python3.12、PyTorch2.10.0+cu128、V10032GB（FP16 VLM／文章符号化・FP32特徴ヘッド＋FP16 autocast）、RTX5070 12GB（BF16生成）で行いました。Irisのソースは固定リビジョンを外部の読み取り専用ソースとして使用します。

```powershell
python scripts/bootstrap.py --vlm
./.venv/Scripts/python.exe scripts/runtime_info.py
```

専用venvを2つ作り、共有PC環境へはインストールしません。`.venv` は Transformers4.57.3、`.venv-vlm` は5.7.0です。モデル本体は含まれていないため、各ライセンスを確認して固定モデルを別途取得します。ダウンロードには数十GBの空き容量が必要です。

```powershell
./.venv/Scripts/python.exe scripts/download_reproduction_assets.py --model iris
./.venv/Scripts/python.exe scripts/download_reproduction_assets.py --model qwen4b
./.venv/Scripts/python.exe scripts/download_reproduction_assets.py --model minicpm
./.venv/Scripts/python.exe scripts/download_reproduction_assets.py --model lfm16
./.venv/Scripts/python.exe scripts/download_reproduction_assets.py --model dino
./.venv/Scripts/python.exe scripts/download_reproduction_assets.py --model clip
```

`--list` はファイル名・容量・ハッシュを表示するだけです。`--verify-only` はダウンロードせず照合します。CLIPの公開取得元リビジョンは記事公開準備時に固定し、使用済みファイルと全3ファイルのSHA-256が一致することを確認しました。DINOとVLMは当初の固定リビジョンを維持しています。LFMは独自ライセンスであり、コードのApache-2.0は適用されません。

## 設定と実行

公開測定を上書きしないよう、新しい出力ディレクトリを指定します。GPU番号は `nvidia-smi -L` の表示に合わせて設定してください。

```powershell
$env:SC_RUN='results/reproduction-001'
$env:SC_FEATURE_GPU='0'
$env:SC_GENERATOR_GPU='1'
./run-text-route.ps1 -Phase Prepare
./run-text-route.ps1 -Phase Caption
./run-text-route.ps1 -Phase Encode
./run-text-route.ps1 -Phase Generate
./run-text-route.ps1 -Phase Refine
./run-text-route.ps1 -Phase NativeFeature
```

入力は公開済み13枚をそのまま照合します。元の8写真の選び直しや新しい評価集合の取得は行いません。VLMは生の説明を保存し、受信側は別プロセスで説明から状態を計算します。生成の本設定は1024px・10ステップ、Refineは元の3写真の第一シードだけ40ステップです。

同じGPUで全工程を使う場合は2つの番号を同じ値にしますが、V100単体では生成器のBF16条件を満たさないため、当初と同じ設定では実行できません。画像生成にはBF16対応CUDA GPUが必要です。標準SDPAへ切り替える場合は `IRIS_ATTENTION_BACKEND=sdpa` を設定し、そのカーネル変更を新しい実験条件として記録します。

モデルパスは `SC_MODEL_QWEN4B`、`SC_MODEL_MINICPM`、`SC_MODEL_LFM16`、評価器は `SC_CLIP_DIR`、`SC_DINO_DIR` で指定できます。既定はリポジトリ内の `models/` です。

公開版の温度監視はnvidia-smiによる読み取りのみです。feature工程70℃、generator工程80℃以上では停止します。元のV100実験は別途、30秒以内のCoolingDesk監視JSON、基板冷却の有効状態、非常状態なし、GPU/HBM70℃未満も確認しました。同じ監視システムを持つ環境では `SC_COOLING_STATE` に既存JSONを指定できます。冷却やファンの設定は変更しません。この監視アダプタとパス設定が公開版での変更点です。

## 評価・レポートの再作成

初回256px特徴方式も含む全459行の評価を再実行する場合は、保存済みの初回pilot結果を参照します。再取得したモデルと新規生成を評価する際、ソース画像は評価器にだけ渡します。

```powershell
./run-text-route.ps1 -Phase Evaluate
./run-text-route.ps1 -Phase Report
```

Evaluateは最初のpilotサムネイルのバイト数取得に除外した特徴ZIPを参照するため、公開版では保存済みencoder-summaryのJPEGコンポーネントbyte記録を使うようにしてあります。新しいpilot自体を再生成するには `./scripts/run-pilot.ps1 -Phase Encode -Run pilot-replay-001`、続いてDecode／Evaluateを実行します。raw特徴・ZIPはその新規ローカルrunだけに保存します。

## 検証範囲

公開準備では保存画像・パケットのハッシュ、集計値、記事リンク、Python構文、量子化の実誤差と実パケットbyteテストを確認します。公開用パス・監視の変更後に、別PCで全推論工程を最初から実行したわけではありません。別GPU・カーネル・ライブラリで画素単位の一致は保証しません。
