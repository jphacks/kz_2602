#!/usr/bin/env python3
"""AtomS3R CAM で撮影 → HTTP APIへ送信 → 結果をOLEDに表示。

先に PC で API サーバーを起動しておく(モデルは起動時に1回だけ読み込まれる):
  cd kz_2602 && uv run python src/main.py --serve --yolo-model models/yolov8n-seg.onnx --coco \
      --host 0.0.0.0 --port 8000

使い方:
  python renpin_bridge.py --node-ip 192.168.1.50 --server http://127.0.0.1:8000 --diameter-cm 24
  # カメラなしで試す:
  python renpin_bridge.py --image sample.jpg --server http://127.0.0.1:8000 --diameter-cm 24

# initial_temperature_c には、既定で MLX90640 の中央表面温度(いま測った表面温度)を渡す。
→ APIの heating_time_s は「その温度から75Cまで」の時間 = 残り加熱時間になる。
   加熱を一時停止して測るたびに実行すれば、残り時間が更新される。
不要なら --no-thermal、手動指定なら --initial-temp 5。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

TARGET_C = 75.0  # API側の目標温度。これ以上の初期温度は400になる


def http_get(url: str, timeout: float = 15.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch_image(node_ip: str, dest: Path) -> None:
    data = http_get(f"http://{node_ip}/capture")
    if len(data) < 1000:
        raise RuntimeError(f"撮影データが小さすぎます({len(data)} bytes)")
    dest.write_bytes(data)


def fetch_thermal(node_ip: str) -> dict | None:
    try:
        return json.loads(http_get(f"http://{node_ip}/thermal", timeout=10))
    except Exception:
        return None


def show_on_oled(node_ip: str | None, l1: str, l2: str = "", l3: str = "") -> None:
    if not node_ip:
        return
    q = urllib.parse.urlencode({"l1": l1, "l2": l2, "l3": l3})
    try:
        http_get(f"http://{node_ip}/show?{q}", timeout=5)
    except Exception as e:  # OLED表示の失敗で全体を止めない
        print(f"warn: OLED表示に失敗: {e}", file=sys.stderr)


def post_estimate(
    server: str,
    image: Path,
    diameter_cm: float,
    initial_temp_c: float | None,
    timeout: float = 120.0,
) -> dict:
    """POST /v1/estimate (multipart/form-data)。仕様: docs/api.md"""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    field("diameter_cm", str(diameter_cm))
    if initial_temp_c is not None:
        field("initial_temperature_c", f"{initial_temp_c:.1f}")
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{image.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + image.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/estimate",
        data=b"".join(parts),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(body)["error"]
            raise RuntimeError(
                f"API {e.code} {err['code']}: {err['message']}"
            ) from None
        except (ValueError, KeyError):
            raise RuntimeError(f"API {e.code}: {body[:200]}") from None


def fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def summarize(result: dict, init_c: float | None) -> tuple[str, str, str]:
    """OLED用の3行(半角のみ)。heating_time_s は『現在の温度 -> 75C』までの時間なので、
    各回の測定で現在温度を initial_temperature_c に渡せば『残り加熱時間』になる。"""
    h, m = result["heating"], result["measurement"]
    l1 = f"Left {fmt_mmss(h['heating_time_s'])} ({h['heating_time_s']:.0f}s)"
    l2 = f"{h['class_id']} W{m['weight_g']:.0f}g H{m['height_cm']:.1f}"
    if h["needs_turning_or_stirring"]:
        l3 = "TURN/STIR!"
    elif init_c is not None:
        l3 = f"now {init_c:.0f}C -> 75C"
    else:
        l3 = "no turning"
    return l1, l2, l3


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--server", default="http://127.0.0.1:8000", help="APIサーバーのURL")
    p.add_argument(
        "--diameter-cm", required=True, type=float, help="ターンテーブル直径[cm]"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--node-ip", help="AtomS3R CAM の IP")
    src.add_argument("--image", type=Path, help="カメラを使わず既存画像を使う")
    p.add_argument(
        "--initial-temp", type=float, help="初期温度[C]を手動指定(thermalより優先)"
    )
    p.add_argument(
        "--no-thermal", action="store_true", help="MLX90640の温度を初期温度に使わない"
    )
    p.add_argument("--save-image", type=Path, default=Path("capture.jpg"))
    p.add_argument("--save-json", type=Path, default=Path("result.json"))
    a = p.parse_args()
    node = a.node_ip

    try:
        health = json.loads(http_get(a.server.rstrip("/") + "/healthz", timeout=5))
        if health.get("status") != "ok":
            raise RuntimeError("サーバーのモデルが未準備です")
    except Exception as e:
        print(f"error: APIサーバーに接続できません({a.server}): {e}", file=sys.stderr)
        show_on_oled(node, "SERVER NG", "start API")
        return 2

    thermal = None
    if node:
        show_on_oled(node, "capturing...")
        fetch_image(node, a.save_image)
        image = a.save_image.resolve()
        if not a.no_thermal:
            thermal = fetch_thermal(node)
    else:
        image = a.image.resolve()

    init = a.initial_temp
    if init is None and thermal:
        init = thermal["center_mean_c"]
    if init is not None and init >= TARGET_C:
        # すでに目標温度(75C)以上: APIは呼ばず「完了」を表示する
        print(f"完了: 表面温度 {init:.1f}C >= {TARGET_C:.0f}C(残り 0:00)")
        show_on_oled(node, "DONE!", f"surface {init:.0f}C", ">= 75C")
        return 0
    if init is not None and init < -30.0:
        print(
            f"warn: 初期温度 {init:.1f}C は範囲外のため送りません(既定20Cを使用)",
            file=sys.stderr,
        )
        init = None

    show_on_oled(
        node,
        "analyzing...",
        f"init {init:.1f}C" if init is not None else "init 20C(def)",
    )
    try:
        result = post_estimate(a.server, image, a.diameter_cm, init)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        show_on_oled(node, "ERROR", str(e)[:21])
        return 1

    if thermal:
        result["surface_thermal_raw"] = thermal
    a.save_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    l1, l2, l3 = summarize(result, init)
    print(
        f"{l1} | {l2} | {l3}"
        + (f" | 現在の温度 {init:.1f}C" if init is not None else "")
    )
    show_on_oled(node, l1, l2, l3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
