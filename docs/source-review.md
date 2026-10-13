# 技術調査の初期整理

確認日: 2026-10-13。ソースは cee0c130a62a9c288ad8918f1f7cdf23cc228d69、公開重みは adc05b909fb05d4276d26aae5a52b36406b2ef26 に固定。

## 取得できる情報と復号に必要な情報

公式 forward の capture 引数で指定ブロック後の画像パッチトークンを取得できる。24層、幅2560、16pxパッチ。先頭8層は画像とテキストを別ストリームで処理し、以降16層は単一ストリーム。REPAの指定層は10。画像側の特徴だけが返るため、途中層から続きを実行するには、その時点のテキスト側状態も別途必要になる。最終層なら後続トランクは不要だが、PiTは noisy RGB を直接受け取る。

ソース確認場所:

- .upstream/iris-3b/src/iris3b/models/dit.py: IrisOutput, forward, features, pixel_embedder(x)
- .upstream/iris-3b/src/iris3b/nn/embeddings.py: PixelEmbedder と PatchEmbedder
- .upstream/iris-3b/src/iris3b/train/trainer.py: REPA用captureと clean image teacher
- .upstream/iris-3b/src/iris3b/downstream/__init__.py: 公開empty_promptの仕様

内部特徴は圧縮済みの符号ではない。16pxパッチのRGB8は768 bytes、一方2560成分のFP16特徴は5120 bytes。実通信ではシリアライズ・量子化パラメータも含めて計測する。

## 関連研究からの設計上の示唆

[Iris-3B](https://arxiv.org/abs/2610.09450) はVAEを使わない生成と下流タスクを研究する。深度・復元で有意な優位を得ていないため、VAEの有無だけで今回の期待結果を決めない。初期実験は正式な1024px品質評価と区別する。

[REPA](https://arxiv.org/abs/2410.06940) は生成モデルの中間表現を外部認識モデルへ整合させる研究。整合した特徴を抽出できても、それを入力する復号器が自動的に存在するとはいえない。DINO系を評価器に選ぶ場合、教師との重なりを評価上の限界として扱う。

[Diffusion Autoencoders](https://arxiv.org/abs/2111.15640) は意味側と確率的細部側の表現を分け、復号器を学習する。Irisの既存層をそのまま通信に流用する場合との違いを整理するための先行研究とする。

[Diffusion-Aided Bandwidth-Efficient Semantic Communication](https://arxiv.org/abs/2510.26442) はテキストと疎な視覚情報、受信側の追加要求を組み合わせる。今回の初期実験では通信路の誤り・フィードバックまでは導入せず、まず送受信に必要な情報量を確定する。

## ライセンス・環境

Irisコード・重みは公式にApache-2.0とされる。派生ヘッド処理は同ソースを参照したため、公開時は同梱NOTICE等を保持する。初期写真はscikit-imageの公式説明でPublic domain/CC0を確認し、出典とハッシュを記録する。生成コードの公開、モデルや写真の再配布、外部サービスへの送信はこの段階では実施しない。

既存GPU環境は読み取り専用依存先として再利用し、追加依存は専用venvに隔離。PyTorch 2.10.0+cu128のsm_70対応は実際のビルド情報で確認済み。完全独立環境は docs/reproduction.md の固定依存を使う。
