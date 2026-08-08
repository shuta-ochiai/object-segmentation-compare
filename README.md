# SAM 3.1 vs YOLO26 比較実験

COCO val2017 のサブセットに対するインスタンスセグメンテーションで、
SAM3.1（プロンプト式・オープン語彙の基盤モデル）とYOLO26-seg（軽量なリアルタイムモデル）の
精度（mask mAP / mIoU）と速度・VRAM使用量を比較する。

## 0. 前提

- GPU: 8GB VRAM 前提（RTX 4060 Laptop で確認）。SAM3.1は軽量版チェックポイントが存在しないため、
  fp16・バッチサイズ1で実行する。
- SAM3.1の重みは Hugging Face `facebook/sam3.1` でゲート付き配布。**事前にアクセス申請が必要。**

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

### 精度 (`results/accuracy_summary.csv`, COCOeval segm)

| モデル | AP | AP50 | AP75 | AR | 予測数 |
|---|---|---|---|---|---|
| YOLO26-seg (nano) | 0.327 | 0.469 | 0.348 | 0.346 | 497 |
| SAM3.1 | 0.507 | 0.775 | 0.558 | 0.564 | 1916 |

### 速度・VRAM (`results/speed_summary.csv`, 20枚)

| モデル | 平均latency | 中央値latency | p95 latency | throughput | ピークVRAM |
|---|---|---|---|---|---|
| YOLO26-seg (nano) | 47.7ms | 22.2ms | 114.7ms | 21.0 img/s | 0.09GB |
| SAM3.1 | 807.1ms | 779.5ms | 1101.8ms | 1.24 img/s | 6.39GB |

YOLO26-segはSAM3.1よりおよそ**17倍高速・71倍省VRAM**。一方SAM3.1のAP/ARが高いのは、
各画像の正解カテゴリ名を事前にテキストプロンプトとして与えている（オラクル情報）ことが
大きく寄与している点に注意（詳細は下記「精度比較の解釈上の注意」を参照）。両者は設計思想・
用途が異なるモデルであり、単純な優劣比較にはならない。

可視化: `results/speed_comparison.png`, `results/accuracy_comparison.png`,
`results/qualitative_examples.png`（マスクの重ね合わせ比較）。

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
- **SAM3.1の速度計測でのOOM**: 全80カテゴリ名を一度にテキストプロンプトとして渡すと、
  grounding attentionの計算量が跳ね上がり8GB VRAMでOOMする（10GB超を要求）。
  `benchmark_speed.py` では精度評価と同様、画像ごとのGT正解カテゴリのみをプロンプトに使うよう
  制限している。
- **精度比較の解釈上の注意**: `benchmark_accuracy.py` / `benchmark_speed.py` のSAM3.1は、
  各画像に写っている正解カテゴリ名を事前にテキストプロンプトとして与えている（オラクル情報）。
  一方YOLO26は80クラス全体から検出しており、この情報を与えられていない。そのためAP/ARの差には
  この非対称性が寄与しており、「SAM3.1の方が高精度」という結論だけを単純に取り出すのは誤り。
