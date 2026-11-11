# bear_det_cpp — 町に出てきた熊を見つける検出器（C++ / WASM）

防犯カメラや玄関カメラの映像から熊を見つける、小さい検出器（yolov8n、1 クラス `bear`）です。
自動撮影カメラの画像集（LILA BC）、観察写真（iNaturalist）、COCO の熊で学習しました。
推論は自前の ONNX ランナ（C++）で、同じ ONNX が C++・Python・ブラウザ（WASM）で同じ枠を出します。

**デモ**: https://yomei-o.github.io/bear_det_cpp/
（米国の住宅地の玄関カメラと、町の図書館の防犯カメラの映像入り。手元の動画・画像、画面共有でも動きます。すべてブラウザの中で計算します）

> **これは研究・お試し用です。** 人の安全にかかわる判断（避難・通報など）には使わないでください。
> 熊がいても鳴らないことも、熊のいない画像で鳴ることもあります（下の「結果」を見てください）。

## 結果

> **いまのモデルの大きな弱点（2026-11-11、再学習中）**: デモの玄関カメラの動画（夜）で、**停めてある黒い車を熊と誤検知し、駐車場を歩く熊はほとんど見つけません**（全 1,014 フレームのうち、車で鳴ったのが 68%、熊で鳴ったのは 4%）。
> 学習の負例が森や草原の自動撮影カメラばかりで、車・家・道路がほとんど入っていなかったためです。また、町なかの熊は全部テスト用に取っておいたので、学習に入っていませんでした。
> 町の負例（COCO の車・犬・人など 6,263 枚、首都高の夜・夕方 750 枚）と、町なかの熊の半分を足して学習し直しています。終わりしだい、モデルと下の数字を差し替えます。

C++ の `bdet val`（デモと同じ前処理）で測りました。警報のしきい値は、デモの初期値と同じ 0.4 です。
「警報」は、画像に 1 つでも熊の枠が出たかどうかです。

| 集まり | モデル | mAP50 | mAP50-95 | 熊の画像で警報が鳴った割合 | 熊のいない画像で誤報が鳴った割合 |
|---|---|---|---|---|---|
| 検証用（自動撮影カメラ・観察写真・COCO、3,263 枚） | 精度優先 960×544 | **0.921** | 0.822 | **90.8%**（1,533 / 1,689） | **5.3%**（84 / 1,574） |
| | 速さ優先 640×352 | 0.900 | 0.793 | 86.5%（1,461 / 1,689） | 5.6%（88 / 1,574） |
| ツキノワグマだけ（137 枚） | 960 | **0.945** | 0.817 | **85.4%**（117 / 137） | — |
| | 640 | 0.929 | 0.805 | 78.8%（108 / 137） | — |
| 町なかテスト（学習に使っていない、1,433 枚） | 960 | **0.872** | 0.750 | **85.8%**（1,230 / 1,433） | — |
| | 640 | 0.835 | 0.699 | 79.1%（1,133 / 1,433） | — |

熊の大きさ（枠の幅が画像の幅の何割か）ごとの、見つかった割合（960、検証用、IoU 0.5 以上）:

| 熊の幅 | 〜5% | 5〜10% | 10〜20% | 20〜35% | 35〜60% | 60%〜 |
|---|---|---|---|---|---|---|
| 見つかった | 20 / 47（43%） | 118 / 179（66%） | 393 / 462（85%） | 455 / 518（88%） | 361 / 386（94%） | 260 / 282（92%） |

わかったこと:

- **ツキノワグマも見つかります。** 学習の熊のほとんどは北米のアメリカクロクマですが、ツキノワグマの mAP50（0.945）は全体（0.921）より高く出ました。ツキノワグマで熊でないものを熊と言ったのは、137 枚で 1 回だけです（適合率 99.2%）。
- **小さい熊は苦手です。** 画像の幅の 5% 未満の熊は半分も見つかりません。遠くの熊を見たいときは、画面を切り出して拡大して入れる必要があります。
- **誤報は熊のいない画像の 5% ほどです。** 鹿・イノシシ・犬・人などが毎日写るカメラでは、このままだと 1 日に何度も鳴ります。実際に使うなら「何コマも続けて熊が出たときだけ鳴らす」などの工夫が要ります。
- **町なかテストの数字は実際より低めです。** 下の「枠の誤り」のとおり、見逃しの中には熊の足跡や糞の写真が多く入っています。
  また、デモのモデルは横長の映像向けの 960×544 なので、縦長のスマホ写真は小さく縮みます（正方形 960×960 に入れる Python の評価では、見逃しは 145 枚でした）。
- **デモの 2 本の動画**では、960 モデルでフレームの約 3 分の 2 で警報が鳴りました（玄関カメラ 68%、図書館 67%、Python で全フレーム）。鳴らないのは主に、熊が画面の外にいる始めと終わりの部分です。

学習: ultralytics 8.4.104、yolov8n.pt から、入力 960、40 epoch、batch 16（GPU で 1.4 時間）。

## 学習データ

| 出どころ | 何に使ったか | 枚数 | 枠 |
|---|---|---|---|
| [LILA BC](https://lila.science/) の自動撮影カメラ（NACTI・SWG・AMMonitor など） | 熊の正例・熊のいない負例 | 正例 6,678、負例 8,528 | MegaDetector v5a の「動物」の枠 |
| [iNaturalist](https://www.inaturalist.org/) の観察写真（CC ライセンスのもの） | ツキノワグマ 385・日本のヒグマ 137・アメリカクロクマ 2,047（学習と検証）、町なかの写真 1,433（テスト専用） | 4,002 | MegaDetector v5a の「動物」の枠 |
| [COCO 2017](https://cocodataset.org/) の熊 | 正例 | 1,009 | 人が付けた枠 |

- **熊の正例**: LILA の熊（アメリカクロクマ、ツキノワグマ、マレーグマ、メガネグマ）は、同じ連続撮影からは 1 枚（珍しい熊は 5 枚まで）にしています。似た画像が並ばないようにするためです。
- **熊のいない負例**: 熊が出るデータセットの、熊以外の動物と空の画像です。イノシシ・犬・鹿・猿・人など、熊と間違えやすいものを多めにしています。
- **枠**: 自動撮影カメラの画像には熊の枠がありません（「この連続撮影に熊がいる」というラベルだけ）。そこで MegaDetector（MIT）に動物の枠を付けさせ、信頼度 0.5 以上の動物が 1 つもない画像は「熊が写っていないコマ」として落としました。
- **分け方**: LILA はカメラの設置場所ごとに 2 割を検証へ（同じ場所の画像が学習と検証の両方に入らないように）。iNaturalist は観察ごと、COCO は元の train / val のままです。
- **町なかテスト**（`test_urban`）: iNaturalist のアメリカクロクマのうち、説明文に「庭・近所・道路・私道・ゴミ・玄関・玄関カメラ・駐車場」などの言葉がある観察です。学習には使っていません。

### 枠の誤り（わかっていること）

MegaDetector は、熊の足跡・糞・掘り返した穴にも「動物」の枠を付けることがあります。iNaturalist の観察には、熊そのものではなく痕跡の写真が多いからです。
町なかテストで警報が鳴らなかった画像を 24 枚見たところ、約半分がこの「熊ではない写真」でした（検出器が鳴らないのが正しい）。
そのため、町なかテストの数字は、実際より低く出ています。

## 前処理（縦横比を保つ）

入力は、グラフが宣言する大きさ（960×544 または 640×352）に、**縦横比を保って縮め、真ん中に置き、余白を灰色（114）で埋めます**。Ultralytics の学習・評価と同じです。
最初の版は縦横比を無視して引き伸ばしていたため、4:3 の自動撮影カメラの写真や縦長のスマホ写真がつぶれ、町なかの熊の見逃しが 3 倍になっていました（145 枚 → 453 枚）。
引き伸ばしは `--stretch` で選べます（C++ と Python の両方）。

## C++ と Python が同じであること

| 何を | 結果 | 道具 |
|---|---|---|
| 推論（同じ ONNX・同じ画像 100 枚。町なか 40、検証 40、ツキノワグマ 20） | 熊の数が 100 枚すべてで一致。対応した枠 79 個の座標の差は中央値 0.06px、99% 点 2.8px、最大 10.5px（大きい写真を 1/k 倍で戻すため、ネットの上の 0.1px が写真の上で数 px になる）。点数の差は最大 0.0065 | `tools/parity_detect.py` |
| ブラウザ（WASM）と Python | 同じ画像で同じ枠（566.6, 676.6, 900.8, 1048.9） | `wasm/test_node.js` |

C++ の学習（`bdet train`）は姉妹リポ [vehicle_det_cpp](https://github.com/yomei-o/vehicle_det_cpp) と同じエンジンで、損失と勾配が ultralytics 8.4.104 と一致することを確かめてあります（`tools/parity_train_det.py`）。
このリポの熊のモデル自体は、Python（ultralytics 8.4.104、GPU）で学習しました。

## 使い方

### 1. 学習データを集める（Python）

```sh
python tools/select_lila.py lila_image_urls_and_labels.csv.zip --out lila      # LILA の熊と負例
python tools/coco_bears.py annotations_trainval2017.zip --out coco             # COCO の熊（注釈だけ落として画像を 1 枚ずつ取る）
python tools/inat_bears.py --out inat                                          # iNaturalist（CC ライセンスのもの、inat.csv に撮影者とライセンス）
python -m megadetector.detection.run_detector_batch MDV5A lila lila_md.json --recursive --output_relative_filenames
python -m megadetector.detection.run_detector_batch MDV5A inat inat_md.json --recursive --output_relative_filenames
python tools/make_dataset.py --lila lila --md lila_md.json --coco coco --inat inat --inat-md inat_md.json --out dataset
```

### 2. 学習する（Python、ultralytics 8.4.104、GPU）

```sh
python tools/train_det.py --data dataset/data.yaml --imgsz 960 --epochs 40 --out models/bear_det.onnx
python tools/train_det.py --export-only <best.pt> --export-hw 352,640 --out models/bear_det_640.onnx   # 軽い版
```

### 3. 推論・評価

```sh
bdet detect --det models/bear_det.onnx --img a.jpg --out a_det.png         # C++
python tools/detect.py --det models/bear_det.onnx --img a.jpg --out a.png   # Python（onnxruntime）
bdet val --det models/bear_det.onnx --data dataset/val --conf 0.4           # mAP と、画像ごとの警報・誤報の割合（C++）
```

他のプログラムからは `bear_det.hpp` だけを使います（`bd::Detector`）。

## ビルド

```sh
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target bdet
sh wasm/build.sh            # ブラウザ用（docs/bdet.js, docs/bdet.wasm）。emsdk が要る
```

## ファイル

- `bear_det.hpp` — 他のプログラムから使う入口（`bd::Detector`）
- `bdet.cpp` — CLI（detect / val / train）
- `pure/` — 推論・学習のエンジン（姉妹リポ [yolo_lpr_cpp](https://github.com/yomei-o/yolo_lpr_cpp) の `pure/` から）
- `tools/` — データ集め、Python の学習・推論、C++ との照合
- `wasm/` — ブラウザ用の入口とビルド
- `docs/` — デモ（GitHub Pages）。モデルと動画を同梱
- `models/` — 学習済みの ONNX

## ライセンス

- 自分のコード（`pure/` のエンジン、`bdet.cpp`、`tools/`、`wasm/`）: BSD-3-Clause
- **モデルの重み**: Ultralytics の `yolov8n.pt` から学習しているので **AGPL-3.0** を引き継ぎます。
- 学習データは**このリポには入れていません**（再配布しません）。
  - LILA BC: データセットごとにライセンスが違います（多くは CDLA-Permissive）
  - iNaturalist: 写真の著作権は撮影者にあります。多くは CC BY-NC（非商用）です。学習に使った写真の一覧（撮影者・ライセンス・URL）は `inat.csv` に残しています
  - COCO: 注釈は CC BY 4.0、画像は Flickr の撮影者ごとのライセンス
  - MegaDetector: MIT
- デモの動画（`docs/`）はどちらも Wikimedia Commons のパブリックドメインです。
  - 玄関カメラ: [Surveillance camera captures bear in Franklin County…](https://commons.wikimedia.org/wiki/File:Surveillance_camera_captures_bear_in_Franklin_County_for_first_time_in_nearly_20_years.webm)（PD-CCTV）
  - 図書館: [Bear inside Mammoth Lakes Library](https://commons.wikimedia.org/wiki/File:Bear_inside_Mammoth_Lakes_Library.webm)（Mono County Office of Education）
  - 日本の熊の、自由に使える動画は見つかりませんでした。
- Eigen（`pure/third_party/eigen_flat/`）: MPL-2.0。stb: MIT / パブリックドメイン
