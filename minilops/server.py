# -*- coding: utf-8 -*-
"""
D4 · 算子服务化 —— "接入工作流"的最后一公里
=========================================
真实产线对应物：算子以服务形态暴露——HTTP 接口 / gRPC / 消息队列消费者。
平台不 import 你的代码，只按 schema 调你。这就是"算子接入协议"。

本文件用 Flask 起一个玩具检测服务：
  POST /detect   {"image": [[r,g,b],...]}        → 检测框 JSON
  GET  /health                                            → 健康检查（产线必备）
  GET  /stats                                              → 运行指标（QPS/延迟/错误数）

三个产线灵魂机制：
  1. 模型只加载一次（挂全局），请求只做推理——不是每次请求 load 模型
  2. /health 与 /stats：可观测性——"算子跑得怎么样"必须有数字
  3. 输入契约校验：坏请求返回 4xx 带原因，不炸进程（一帧坏数据不能挂整条线）

跑法：
  python -X utf8 minilops/server.py
  curl http://127.0.0.1:8790/health
  curl -X POST http://127.0.0.1:8790/detect -H "Content-Type: application/json" ^
       -d "{\"image\": [[10,10,10],[240,240,10],[10,10,10]]}"
"""

import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, request

from preprocess import preprocess
from operator_core import DetectorCore, decode, nms

app = Flask(__name__)

# ── 产线算子的第一铁律：模型加载一次，常驻进程 ──
CORE = DetectorCore(kernel=np.array([[1.9, 1.9], [1.9, 1.9]]), bias=-3.2, version="0.2-demo")

# ── 可观测性：最简指标收集 ──
STATS = {"n_req": 0, "n_err": 0, "latencies": deque(maxlen=100), "started": time.time()}


@app.post("/detect")
def detect():
    """同步推理接口。schema：{image: H×W×3 数组} → {boxes, version, n}"""
    t0 = time.perf_counter()
    STATS["n_req"] += 1
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        STATS["n_err"] += 1
        return jsonify({"error": "missing field: image (expect H×W×3 array)"}), 400
    try:
        img = np.asarray(payload["image"], dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image shape {img.shape} 不是 (H, W, 3)")
        x = preprocess(img)                 # 预处理（D1）
        hm = CORE.forward(x)                # 推理（D2）
        boxes = nms(decode(hm, thr=0.6))    # 后处理（D2）
        out = {
            "n": len(boxes),
            "boxes": [{"x": b.x, "y": b.y, "score": round(b.score, 3)} for b in boxes],
            "model_version": CORE.version,   # 可追溯：框是谁标的
        }
        STATS["latencies"].append((time.perf_counter() - t0) * 1000)
        return jsonify(out)
    except (ValueError, TypeError) as e:
        STATS["n_err"] += 1
        return jsonify({"error": f"bad input: {e}"}), 400
    except Exception as e:   # 未知异常也不许挂进程
        STATS["n_err"] += 1
        return jsonify({"error": f"internal: {type(e).__name__}"}), 500


@app.get("/health")
def health():
    """健康检查：平台探活用。"""
    return jsonify({"ok": True, "version": CORE.version, "uptime_s": round(time.time() - STATS["started"], 1)})


@app.get("/stats")
def stats():
    """运行指标：平均/P95 延迟、请求数、错误率。真实产线接 Prometheus。"""
    lat = list(STATS["latencies"])
    return jsonify({
        "n_req": STATS["n_req"],
        "n_err": STATS["n_err"],
        "latency_ms": {
            "avg": round(np.mean(lat), 2) if lat else None,
            "p95": round(np.percentile(lat, 95), 2) if lat else None,
        },
    })


if __name__ == "__main__":
    print("MiniLOps 检测算子服务 → http://127.0.0.1:8790")
    print("试: curl http://127.0.0.1:8790/health")
    print('试: curl -X POST http://127.0.0.1:8790/detect -H "Content-Type: application/json" '
          '-d "{\\"image\\": [[10,10,10],[240,240,10],[10,10,10]]}"')
    app.run(host="127.0.0.1", port=8790, debug=False)
