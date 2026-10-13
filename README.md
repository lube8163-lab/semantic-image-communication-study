# Semantic image communication: features vs. text

[日本語のまとめ記事](https://lube8163-lab.github.io/semantic-image-communication-study/) · [記事のMarkdown](articles/semantic-image-communication-study.md) · [全条件の画像レポート](results/text-route-20261013/report.html) · [画像の出典](ASSETS.md)

Exploratory, untrained research harness comparing frozen Iris-3B intermediate-feature replay with image → VLM text → image reconstruction. This is a small local feasibility study, not a trained compression codec or a held-out benchmark.

写真11枚では、Qwen3-VL-4Bの詳細文は平均445BでCLIP画像類似度0.853。容量上限を揃えた縮小WebPは平均409BでCLIP 0.676、PSNRではWebPが21.48dB、文章再生成が10.58dBでした。凍結ヘッドによる中間特徴＋JPEGは、サムネイル単体を通信効率で上回る結果を示せませんでした。文章で場面は伝わっても、個体・顔・配置・形状の復元には限界があります。

## What is included

- Research code, fixed model revisions/SHA-256 manifests and the original protocol.
- 13 licensed/attributed normalized inputs, 78 raw VLM captions, exact UTF-8 text packets.
- All 156 main generations, 2 empty controls and 18 40-step checks at native1024 and evaluated256 resolution.
- 459 measurement rows, descriptive paired image bootstrap summaries, JPEG/WebP baselines, feature zero/shuffle controls and diagnostic failures.
- Japanese article, comparison reports and a static GitHub Pages builder.

Model weights, external source checkout, dependencies, environments, high-dimensional feature/text caches, binary feature ZIPs and raw workstation logs are excluded. This repository is approximately 315 MB before Git compression, mostly PNG outputs. Model sharing is assumed; per-image payload excludes initial model distribution and transport overhead.

## Inspect results without a GPU

Read the article or download the repository and open `results/text-route-20261013/report.html`. Recompute the recorded numeric summaries and verify image/text-packet hashes with Python 3.12:

```powershell
python scripts/verify_publication.py
```

The verifier uses only Python's standard library. The article and site can be rebuilt with:

```powershell
python -m pip install -r requirements-site.txt
python scripts/build_site.py
python -m http.server 8765 --directory site
```

## Reproduce inference

See [the public reproduction guide](docs/text-route-reproduction.md). It distinguishes published-result verification from fresh inference. Download models separately according to their licenses. New runs use a separate `SC_RUN`, so the published measurements remain intact.

The experiment used Windows/Python3.12, PyTorch2.10+cu128, V10032GB for FP16 VLM/text encoding and RTX5070 for BF16 native1024 sampling. The public export replaces workstation paths and GPU IDs with configurable settings and uses a portable read-only monitoring adapter. It has not been fully rerun on a clean second PC. Frozen model equations, text masks, precision choices and measured outputs were not changed for publication.

## Limitations

11 convenience photos and 2 synthetic controls; unknown pretrained overlap; no new decoder training; no real communication channel; 10 sampling steps rather than Iris's recommended100; 40-step checks on3 photos only. CLIP/DINO image cosines do not prove identity/counts/layout. Bootstrap intervals are descriptive and unadjusted for multiple comparisons. DINO has a family overlap with Iris's REPA design. Raw caption fact checks are limited observations, not blinded human evaluation.

## Licensing and provenance

Research code is Apache-2.0. Original article, numeric charts and tables are CC BY4.0. External photographs and reference captions keep their own licenses; see [ASSETS.md](ASSETS.md) and [NOTICE](NOTICE). No model weights are redistributed. Source-file hashes from the measured implementation and the public-export policy are recorded in [sources/export-provenance.json](sources/export-provenance.json).
