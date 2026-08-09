# object-segmentation-compare

COCO val2017 のサブセットに対するインスタンスセグメンテーションで、複数のモデル・サイズの
精度（mask mAP）と速度・VRAM使用量を比較する。現在対象にしているのは:

- **YOLO26-seg** (n/m/x) — 軽量・閉集合（COCO80クラス）のリアルタイムモデル
- **SAM3.1** — プロンプト式・オープン語彙の基盤モデル（テキストプロンプトで概念を指定）
- **RF-DETR-Seg** (Nano/Medium/Large) — DETRベースのリアルタイム検出・セグメンテーションモデル

今後さらにモデルを追加する前提でリポジトリ名を汎用化している。追加する場合は `models.py` に
同じ出力形式（COCO形式の予測リストを返す関数）でラッパーを追加し、`benchmark_accuracy.py` /
`benchmark_speed.py` の `MODELS` リストに登録すればよい。

## 0. 前提

- GPU: 8GB VRAM 前提（RTX 4060 Laptop で確認）。SAM3.1は軽量版チェックポイントが存在しないため、
  fp16・バッチサイズ1で実行する。
- SAM3.1の重みは Hugging Face `facebook/sam3.1` でゲート付き配布。**事前にアクセス申請が必要。**
- YOLO26・RF-DETRの重みはアクセス申請不要で初回実行時に自動ダウンロードされる。

## 1. 環境構築

```bash
cd ~/object-segmentation-compare
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. SAM3.1 へのアクセス申請（承認待ちが発生する。数時間〜かかることがある）

1. https://huggingface.co/facebook/sam3.1 を開き、Request access を送信
2. 承認メールが来たら、HFトークンを発行 (https://huggingface.co/settings/tokens)
3. ログイン: `hf auth login`
4. チェックポイントのファイル一覧を確認し、`models.py` 内の `SAM3_CHECKPOINT` 変数を
   実際のファイル名に合わせて調整する（ゲート中は一覧を閲覧できないため、承認後に確認する）
5. ダウンロード:
   ```bash
   huggingface-cli download facebook/sam3.1 --local-dir ./checkpoints
   ```

YOLO26の重みは初回実行時に自動ダウンロードされる（アクセス申請不要）。

## 3. データセット準備

```bash
python download_data.py
```

`data/subset_accuracy/`（精度評価用、約100枚）と `data/subset_speed/`（速度評価用、20枚）が作成される。
SAM3.1は公開ベンチマークでハイエンドGPU上でも約2.9秒/画像かかっており、ノートPC GPUではさらに
遅くなる可能性があるため、速度評価のサンプル数は少なめにしている。

## 4. 実行順序

```bash
# HFアクセス承認前でも実行可能（YOLO26のみ動作確認）
python -c "from models import run_yolo; run_yolo('data/subset_accuracy')"

# HFアクセス承認・チェックポイント配置後
python benchmark_accuracy.py
python benchmark_speed.py
python visualize.py
```

結果は `results/` 以下にCSV/JSON/PNGとして出力される。

## 結果（COCO val2017サブセット, RTX 4060 Laptop 8GB）

### 精度 (`results/accuracy_summary.csv`, COCOeval segm, 精度評価用100枚)

YOLO26・RF-DETRは`conf=0.001`（標準的なmAP評価プロトコルに近い、ほぼ無フィルタ）で計測。
SAM3.1は`conf=0.25`のまま（テキストプロンプト方式のため影響は小さい）。

| モデル | AP | AP50 | AP75 | AR | 予測数 |
|---|---|---|---|---|---|
| YOLO26-seg (n) | 0.391 | 0.601 | 0.402 | 0.509 | 16383 |
| YOLO26-seg (m) | 0.486 | 0.751 | 0.508 | 0.600 | 12808 |
| YOLO26-seg (x) | 0.512 | 0.771 | 0.540 | 0.617 | 10691 |
| SAM3.1 | 0.507 | 0.775 | 0.558 | 0.564 | 1916 |
| RF-DETR-Seg (Nano) | 0.426 | 0.650 | 0.445 | 0.514 | 10000 |
| RF-DETR-Seg (Medium) | 0.476 | 0.720 | 0.507 | 0.603 | 20000 |
| RF-DETR-Seg (Large) | 0.488 | 0.715 | 0.534 | 0.620 | 20000 |

`conf=0.25`で事前フィルタしていた旧結果より全モデルでAP/ARが明確に改善した（特にAR）。
これは標準的なCOCO mAP評価がconfidenceでほぼ絞り込まずCOCOeval側にPR曲線全体を計算させる
のに対し、`conf=0.25`という「サービング用のしきい値」で先に足切りすると低confidence側の
真陽性を再現率から失ってしまうため（詳細は下記「実装時に判明した注意点」を参照）。

### 速度・VRAM (`results/speed_summary.csv`, 速度評価用20枚)

| モデル | 平均latency | 中央値latency | p95 latency | throughput | ピークVRAM |
|---|---|---|---|---|---|
| YOLO26-seg (n) | 47.7ms | 22.2ms | 114.7ms | 21.0 img/s | 0.09GB |
| YOLO26-seg (m) | 46.8ms | 29.5ms | 90.1ms | 21.3 img/s | 0.28GB |
| YOLO26-seg (x) | 59.5ms | 45.8ms | 101.4ms | 16.8 img/s | 0.58GB |
| SAM3.1 | 807.1ms | 779.5ms | 1101.8ms | 1.24 img/s | 6.39GB |
| RF-DETR-Seg (Nano) | 58.7ms | 53.6ms | 81.4ms | 17.0 img/s | 0.21GB |
| RF-DETR-Seg (Medium) | 61.0ms | 58.6ms | 82.3ms | 16.4 img/s | 0.29GB |
| RF-DETR-Seg (Large) | 75.5ms | 78.0ms | 94.0ms | 13.2 img/s | 0.33GB |

**読み方の注意:**
- SAM3.1は他モデルよりおよそ**14〜17倍遅く、20〜70倍のVRAM**を使う。それでもAP単体では
  YOLO26-seg (x)（0.512）がSAM3.1（0.507）とほぼ並ぶか僅かに上回った。SAM3.1には各画像の
  正解カテゴリ名を事前にテキストプロンプトとして与えている（オラクル情報）にもかかわらず、
  ヒントなしで80クラス全体から検出しているYOLO26-xがAPで並んだのは注目に値する
  （AP50/AP75ではSAM3.1がまだ上）。
- YOLO26とRF-DETR-Segは同じ「ヒントなしで80クラスから検出」という条件なので、この2つの間の
  AP/AR差はフェアな比較。修正後の数値では、最小サイズ（n/Nano）同士はRF-DETR-Segが優勢だが、
  中間・大サイズ（m/Medium, x/Large）ではYOLO26-segが逆転してRF-DETR-Segを上回った。
  `conf=0.25`のときの「RF-DETR-Segが一貫して優勢」という結論は、しきい値設定の副作用だった
  ことが分かる。
- モデルサイズを上げても精度が単調に伸びるとは限らない（例: RF-DETR-Seg MediumとLargeのAPは
  ほぼ同値）。100枚という小さいサブセットでの結果なので、この程度の差はノイズの可能性がある。

可視化: `results/speed_comparison.png`, `results/accuracy_comparison.png`,
`results/qualitative_examples.pdf`（入力画像・YOLO26-seg(n)・SAM3.1・RF-DETR-Seg(Nano)を並べた
マスク比較、全98画像・20ページ）。

## トラブルシューティング

- `'SimpleTokenizer' object is not callable` エラー: `pip uninstall clip -y && pip install git+https://github.com/ultralytics/CLIP.git`
  （初回実行時は自動でインストールされる）
- SAM3.1実行時にOOMする場合: `benchmark_speed.py` / `benchmark_accuracy.py` は都度
  `torch.cuda.empty_cache()` を呼んでいるが、それでも足りない場合は他のGPUプロセスを終了するか、
  画像を1枚ずつ別プロセスで実行する

## 実装時に判明した注意点

- **チェックポイントファイル名**: `facebook/sam3.1` の実際の重みファイル名は `sam3.1_multiplex.pt`
  （`sam3.1.pt` ではない）。ultralyticsはファイル名に `"sam3"` が含まれるかでSAM3系列を判定するため、
  そのまま `checkpoints/` に置けば動作する。
- **SAM3.1の呼び出し方**: `SAM("sam3.1_multiplex.pt").predict(text=[...])` は `text` が
  無効な引数としてエラーになる。`ultralytics.models.sam.SAM3SemanticPredictor` を使い、
  `predictor.set_image(path)` → `predictor(text=[...])` の順で呼ぶ必要がある
  （`models.py` の `run_sam31` / `benchmark_speed.py` の `benchmark_sam31` を参照）。
- **`retina_masks=True` が必須**: YOLO26の `model.predict()` はデフォルトだと `masks.data` が
  ストライド調整後の推論解像度（例: 448×640）のままで、元画像解像度（例: 427×640）に
  リサイズされない。これに気づかずCOCOevalに投入すると、マスクのサイズ不一致でIoUがほぼ0になり
  YOLO26のAPが実際より大幅に低く出る（実測: 修正前AP=0.099 → 修正後AP=0.327）。
- **精度評価用のconfidenceしきい値は下げる**: 当初`conf=0.25`（サービング用の実用的なしきい値）
  で精度評価もしていたが、これはCOCOevalが内部でPR曲線を計算する前提を崩し、低confidenceの
  真陽性を再現率から失わせてAP/ARを実際より低く見せてしまう（`benchmark_accuracy.py`は
  `EVAL_CONF=0.001`に変更、`benchmark_speed.py`は実運用に近い`conf=0.25`のまま据え置き）。
  実測でも`conf=0.25`→`0.001`でYOLO26-seg(n)のAR=0.346→0.509、AP=0.327→0.391と大きく改善し、
  「RF-DETR-Segの方が常にYOLO26より高精度」という`conf=0.25`時点の結論も一部覆った
  （中〜大サイズではYOLO26が逆転）。
- **SAM3.1の速度計測でのOOM**: 全80カテゴリ名を一度にテキストプロンプトとして渡すと、
  grounding attentionの計算量が跳ね上がり8GB VRAMでOOMする（10GB超を要求）。
  `benchmark_speed.py` では精度評価と同様、画像ごとのGT正解カテゴリのみをプロンプトに使うよう
  制限している。
- **精度比較の解釈上の注意**: `benchmark_accuracy.py` / `benchmark_speed.py` のSAM3.1は、
  各画像に写っている正解カテゴリ名を事前にテキストプロンプトとして与えている（オラクル情報）。
  一方YOLO26・RF-DETRは80クラス全体から検出しており、この情報を与えられていない。そのため
  SAM3.1のAP/ARの高さにはこの非対称性が寄与しており、「SAM3.1の方が高精度」という結論だけを
  単純に取り出すのは誤り。
- **RF-DETRのimport path**: `rfdetr.util.coco_classes` ではなく `rfdetr.assets.coco_classes`
  （`COCO_CLASSES`は`{category_id: name}`のdictで、COCOの公式category_id採番と一致する）。
  モデルクラスは `RFDETRSeg{Nano,Small,Medium,Large,XLarge,2XLarge}`。
- **RF-DETRはmask解像度の罠がない**: YOLO26と違い、`model.predict(image, threshold=...)`が
  返す`detections.mask`はデフォルトで元画像解像度になっている（実測で確認済み）ので
  `retina_masks`のような追加対応は不要。
- **RF-DETRの速度計測は意図的にfp32/eagerのまま**: `model.inference(compile=True,
  dtype=torch.float16)`でJITコンパイル+fp16の最適化ができるが、他モデルと条件を揃えるため
  あえて使っていない。有効にすればRF-DETRはさらに速くなる見込み。
- **RF-DETRのモデルロード時の警告**: `_kp_active_mask`など一部パラメータがチェックポイントに
  無くランダム初期化される旨の警告が出るが、これはkeypoint検出用のパラメータでセグメンテーション
  タスクには影響しない（upstream側の既知の状態）。
- **GPUの競合に注意**: 他プロセス（別のClaude Codeセッション等）が同じGPUを使っていると、
  速度計測（latency/throughput/VRAM）が不当に悪化する。`nvidia-smi`で他プロセスが無いことを
  確認してから`benchmark_speed.py`を実行すること。精度（AP/AR）自体はGPU競合の影響を受けない
  （遅くなるだけで計算結果は変わらない）。
