# HTTP API仕様

## 1. 概要

食品画像から、ターンテーブルを基準にした食品の寸法・重量・加熱パラメーターを推定するHTTP APIです。

サーバーは起動時に食品セグメンテーション、深度推定、遮蔽物検出のモデルを一度だけ読み込みます。
モデルの設定はリクエストでは変更できません。

- ベースURL: `http://127.0.0.1:8000`
- APIバージョン: `v1`
- 入力形式: `multipart/form-data`
- 成功レスポンス形式: `application/json`
- OpenAPI UI: `/docs`
- OpenAPI JSON: `/openapi.json`

## 2. サーバー起動

```bash
uv run python src/main.py \
  --serve \
  --yolo-model path/to/food.onnx \
  --class-names rice meat vegetable bread \
  --any-object \
  --depth-model models/depth-anything-v2-small.onnx \
  --host 127.0.0.1 \
  --port 8000
```

`--coco` を使う場合は `--class-names` と併用できません。モデル設定はサーバー起動時に固定され、初期構成では1プロセス・1ワーカーで動作します。

## 3. エンドポイント

### `GET /healthz`

モデルの準備状態を確認します。

#### 成功: `200 OK`

```json
{
  "status": "ok"
}
```

#### 未準備: `503 Service Unavailable`

```json
{
  "error": {
    "code": "not_ready",
    "message": "models are not ready"
  }
}
```

モデルの初期化に失敗した場合も、プロセスは起動したまま未準備状態になり、このレスポンスを返します。

### `POST /v1/estimate`

1枚の食品画像を解析します。初版は1画像につき1食品の検出を想定しています。

#### リクエスト

`Content-Type: multipart/form-data`

| フィールド | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| `image` | file | はい | OpenCVでデコード可能な画像。拡張子は判定に使用しません |
| `diameter_cm` | string形式の数値 | はい | ターンテーブルの実直径。正の有限値 |
| `initial_temperature_c` | string形式の数値 | いいえ | 加熱対象物の初期温度（℃）。有限値かつ75℃未満。省略時は20℃ |

例:

```bash
curl -X POST http://127.0.0.1:8000/v1/estimate \
  -F image=@path/to/input.jpg \
  -F diameter_cm=20 \
  -F initial_temperature_c=5
```

#### 成功: `200 OK`

CLIのJSON出力と同じ構造を返します。

```json
{
  "calibration": {
    "center_px": [320.0, 240.0],
    "major_axis_px": 500.0,
    "minor_axis_px": 350.0,
    "angle_deg": 0.0,
    "diameter_cm": 20.0,
    "alpha_cm_per_px": 0.04,
    "theta_rad": 0.8
  },
  "detection": {
    "class_id": "rice",
    "confidence": 0.95,
    "y_base_px": 400
  },
  "material": {
    "class_id": "rice",
    "density_g_per_cm3": 1.05,
    "epsilon_prime": 40.0,
    "epsilon_double_prime": 12.0,
    "tan_delta": 0.3,
    "specific_heat_j_per_g_k": 2.5
  },
  "measurement": {
    "top_x_px": 320,
    "top_y_px": 250,
    "height_cm": 2.0,
    "area_cm2": 100.0,
    "volume_cm3": 133.3,
    "weight_g": 140.0
  },
  "heating": {
    "class_id": "rice",
    "penetration_depth_cm": 1.5,
    "needs_turning_or_stirring": true,
    "required_energy_j": 19250.0,
    "heating_time_s": 55.0
  }
}
```

数値は入力画像と推論結果によって変わります。上記はレスポンス構造を示す例です。

## 4. エラー

エラーは次の形式に統一します。

```json
{
  "error": {
    "code": "invalid_diameter",
    "message": "diameter_cm must be a positive finite number"
  }
}
```

| HTTPステータス | `code` | 条件 |
| --- | --- | --- |
| `400` | `missing_image` | `image` が指定されていない |
| `400` | `invalid_image` | 空ファイル、または画像をデコードできない |
| `400` | `invalid_diameter` | `diameter_cm` が欠落、数値でない、0以下、または有限値でない |
| `400` | `invalid_initial_temperature` | `initial_temperature_c` が数値でない、有限値でない、または目標温度の75℃以上 |
| `413` | `image_too_large` | ファイルが10 MiBを超える、または画像の縦横が4096pxを超える |
| `422` | `pipeline_error` | ターンテーブル検出、食品検出、深度推定などに失敗 |
| `503` | `not_ready` | モデルが準備できていない |
| `500` | `internal_error` | 想定外のサーバーエラー |

`500` の場合、内部例外の詳細はレスポンスに含めません。詳細はサーバーログを確認してください。

## 5. 制限と運用上の注意

- 画像アップロードの上限は10 MiBです。
- 画像の縦または横の上限は4096pxです。
- 1リクエストにつき1画像を処理します。
- 推論処理は同期的に実行されます。高負荷時はリバースプロキシやキューの導入を検討してください。
- ワーカーを増やすと、各プロセスでモデルが個別にロードされます。メモリ使用量を確認してから変更してください。
- 認証、TLS、CORS、レート制限、画像URL入力、base64 JSON入力は初版のAPIに含みません。本番公開時はリバースプロキシなどで追加してください。
