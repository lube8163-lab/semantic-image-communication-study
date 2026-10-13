# 画像・データの出典と利用条件

本リポジトリのコード用Apache-2.0ライセンスは、外部の写真や参照文のライセンスを置き換えません。

入力画像にはRGB変換、中央の正方形クロップ、Lanczosによる256×256への縮小を行いました。図では再縮小・配置・ラベル追加を行っています。通常コーデックの出力はJPEG/WebP符号化と拡大による変更を含みます。特徴ヘッドの出力は量子化・凍結ヘッドによる変更です。文章経由の出力はAIによる新たな生成画像として明示します。下記の原写真の権利者が実験や生成結果を支持しているという意味ではありません。

| 入力ID | 作者／クレジット | 題名／原ページ | ライセンス |
|---|---|---|---|
| astronaut | NASA / Eileen Collins | [astronaut](https://scikit-image.org/docs/stable/api/skimage.data.html#astronaut) | Public domain |
| coffee | Rachel Michetti / Pikolo Espresso Bar | [coffee](https://scikit-image.org/docs/stable/api/skimage.data.html#coffee) | CC0 |
| chelsea | Stefan van der Walt | [chelsea](https://scikit-image.org/docs/stable/api/skimage.data.html#chelsea) | CC0 |
| red_circle_left | lube8163-lab / PIL synthetic control | [red_circle_left](scripts/prepare_assets.py) | CC0 |
| blue_square_right | lube8163-lab / PIL synthetic control | [blue_square_right](scripts/prepare_assets.py) | CC0 |
| 07cdb16d66edeacb | Michael Van Gorkom | [mlv_VanGorkom417-r](https://www.flickr.com/photos/vangorkomm/3160601139) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| 67e67c811336e94b | Chika Watanabe | [Hiking Trail from Curry Village](https://www.flickr.com/photos/chikawatanabe/39423671) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| 79083da149309142 | Paul Sturgess | [Birdie](https://www.flickr.com/photos/odetothebigsea/3277383288) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| f5f2decfd4122750 | Karsun Designs | [DSC_0171](https://www.flickr.com/photos/karsund/6187601309) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| a158e89a4415eb57 | Stephen Michael Barnett | [Hughs new Polo](https://www.flickr.com/photos/httpwwwflickrcomphotostopend/2203559148) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| e84c2f66a5963a9a | The Secret History GRSG | [Tonight is: #popcorn + #crochet hook 4 #dreadlocks](https://www.flickr.com/photos/rmdemsick/7640668976) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| 05430da00e88b5f9 | c-66 | [Untitled](https://www.flickr.com/photos/c-66/509197136/) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| 56e803d5e8c580e0 | charlie vinz | [kindergarten](https://www.flickr.com/photos/vinzcha/3856950504) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |

Open Images由来8枚のライセンス・作者・題名は、固定された公式の画像メタデータCSVから記録しました。写真自体はデータセットの公開ミラーから取得しました。原写真を加工したパネルとコーデック復元には各原ライセンスを維持してください。

メタデータ: https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv

ライセンス説明: https://storage.googleapis.com/openimages/web/download_v7.html

参照文: Localized Narratives, Pont-Tuset et al., Connecting Vision and Language with Localized Narratives, ECCV 2020. [プロジェクト](https://google.github.io/localized-narratives/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 注釈者の参照文は評価用にだけ保持し、VLM・生成器へ送っていません。

数値のみの独自グラフはCC BY 4.0、比較画像中の元写真は上記各ライセンスです。生成結果はAIの実験出力であり、元の人物や出来事の記録として用いるものではありません。
