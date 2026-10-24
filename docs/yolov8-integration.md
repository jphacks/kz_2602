# YOLOv8セグメンテーション実装メモ

## 対応モデル

食品領域の抽出には、エクスポート済みのYOLOv8n-seg ONNXモデルを使用する。
モデルファイルはリポジトリへ含めず、ライセンスと配布条件を確認したうえで
`models/food.onnx` などに配置する。

## 設定

`src/main.py` の `create_food_segmenter` にJSON互換の設定を渡して、セグメンターを
構築できる。

```python
from main import create_food_segmenter

food_segmenter = create_food_segmenter(
    {
        "model_path": "models/food.onnx",
        "model_pt_path": "models/food.pt",
        "model_pt_url": "https://example.com/models/food.pt",
        "model_url": "https://example.com/models/food.onnx",
        "model_sha256": "<64-character-hexadecimal-sha256>",
        "class_names": ["rice", "meat", "vegetable", "bread"],
        "input_size": [640, 640],
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
    }
)
```

`class_names` はモデルのクラスインデックス順と一致させる。物性データベースで使用
できるクラスは現在 `rice`、`meat`、`vegetable`、`bread` である。

`model_path` が存在しない場合は、`model_url` からONNXファイルを起動時にダウンロード
する。ダウンロードは一時ファイルへ行ってから配置するため、途中で失敗したファイルは
モデルとして残らない。`model_sha256` を指定すると、既存ファイルとダウンロードした
ファイルのSHA-256を検証する。`model_path` がすでに存在する場合はダウンロードせず、
指定されたハッシュだけを検証する。

`model_pt_path` を指定すると、ONNXが存在しない場合にUltralyticsで `.pt` モデルを
CPU上でONNXへ変換する。`.pt` も存在しない場合は `model_pt_url` から取得する。
変換時は `nms=False` を指定し、後処理を `FoodSegmenter` 側で行う。変換機能を使う
場合は、実行環境に `ultralytics` をインストールする。

```bash
uv add ultralytics
```

## 入出力契約

`FoodSegmenter.predict` はBGRの `uint8` 画像を受け取る。内部でletterbox、RGB変換、
正規化を行ってからONNX Runtimeで推論する。パイプラインの公開入力はRGBであり、
`run_pipeline` 内部でBGRへ変換される。

対応するONNX出力は、YOLOv8-segのpredictionとprototypeの2テンソルである。

* predictionは `4 + クラス数 + マスク係数数` チャンネルを持つ。
* prototypeはマスク係数と空間特徴を持つ。
* channel-firstとchannel-lastの一般的な軸を受け付ける。
* 信頼度閾値処理とNMSの後、検出が1件だけ残ることを要求する。
* 出力マスクは元画像サイズのbool配列である。

返却される `FoodDetection` には、マスク、設定されたクラス名、信頼度、マスク最下部
の `y_base_px` が含まれる。

## 検証

モデルファイルがない環境では、FakeSessionを使ったユニットテストで前処理、出力軸、
マスク復元、NMS、設定検証を確認できる。

```bash
uv run pytest tests/test_food_segmentation.py tests/test_main.py
```

実モデルを使う場合は、代表画像でマスク位置、クラス名、信頼度、`y_base_px`、推論時間を
確認する。モデルのクラス順と設定の `class_names` が一致しない場合、後段の物性参照結果も
誤るため注意する。
