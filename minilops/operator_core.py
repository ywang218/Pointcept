# -*- coding: utf-8 -*-
"""
D2 · 算子内核 + 计算图雏形 —— "模型"本体
=========================================
真实产线对应物：训好的 2D 检测模型（YOLO/DETR 家族），
通常已导出成 ONNX：一张 compute graph（节点=算子，边=张量），由推理引擎执行。

这里做玩具版，但**保住三个真东西**：

1. DetectorCore：一个真会"训练"的极简检测器
   - 结构：单通道 2×2 卷积（4 个权重）→ bias → 逐点 sigmoid 得到热力图
   - 语义：热力图高分的位置 = "有目标"（红绿灯玩具版）
   - 手写了 forward 和 backward（数值梯度同款公式），
     retrain_feedback.py 会真的用它做梯度下降——不是假装训练。

2. graph_forward(ops)：用 list[dict] 模拟 ONNX 计算图
   - 每个 dict = 一个图节点：{op: 类型, inputs: [...], weights: {...}, out: 名字}
   - 张量放在 dict 里按名字传递 = ONNX 的 initializer / intermediate value
   - 真实 ONNX 里这个 dict 是 protobuf 节点，执行引擎（ORT）做的事
     和这个 for 循环本质相同：拓扑序遍历 + 查表取输入 + 调 kernel。
   - E2 实验（装 torch 后）导出真 ONNX 时，回头对照这个结构。

3. decode + NMS：检测算子的"后处理"段
   - 热力图 → 候选框 → 按分数过滤 → NMS 去重叠
   - NMS 是 2D 检测部署的经典后处理：图形学的 hi-z/遮挡剔除近亲
     （都是"按重要性排序，贪心保留，抑制邻居"）。

权重序列化：真实产线是 .onnx/.pt 文件；玩具版是 json dict。
版本号：算子必须可追溯——"这帧是谁标的"= 哪个版本的权重。
"""

import json
from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────── 1. 检测器内核 ───────────────────────────

@dataclass
class DetectorCore:
    """极简 2D 检测"模型"：2×2 conv → bias → sigmoid 热力图。

    weights 是 dict 而不是矩阵，是为了让"加载/保存权重"像真的 checkpoint。
    """
    version: str = "0.1.0"
    kernel: np.ndarray = field(default_factory=lambda: np.random.default_rng(7).normal(0, 0.1, (2, 2)))
    bias: float = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(H, W) 归一化图 → (H-1, W-1) 热力图。手写 2×2 卷积（无 padding）。"""
        if x.ndim != 2:
            raise ValueError(f"期望 (H, W)，得到 {x.shape}")
        h, w = x.shape
        if h < 2 or w < 2:
            raise ValueError("图太小，卷不动")
        # im2col 思想的迷你版：把四个邻居摆成矩阵再点积
        tiles = np.stack([
            x[0:h-1, 0:w-1], x[0:h-1, 1:w],
            x[1:h,   0:w-1], x[1:h,   1:w],
        ], axis=-1)                                    # (H-1, W-1, 4)
        z = tiles @ self.kernel.reshape(-1) + self.bias  # (H-1, W-1)
        return 1.0 / (1.0 + np.exp(-z))                # sigmoid

    def loss_grad(self, x: np.ndarray, target: np.ndarray):
        """前向 + 反向。target 是 (H-1, W-1) 的 0/1 标签（打回修正后的真值）。

        返回 (loss, grad_kernel, grad_bias)。
        loss 用 BCE——把"热力图每个位置是不是目标"当二分类。
        手写这 20 行 = 你已经会"训练"的最小闭环：
        forward → 和标签比 → 链式法则求梯度。
        """
        h, w = x.shape
        tiles = np.stack([
            x[0:h-1, 0:w-1], x[0:h-1, 1:w],
            x[1:h,   0:w-1], x[1:h,   1:w],
        ], axis=-1)
        z = tiles @ self.kernel.reshape(-1) + self.bias
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-7
        loss = -np.mean(target * np.log(p + eps) + (1 - target) * np.log(1 - p + eps))
        dz = (p - target) / target.size               # BCE 对 z 的梯度（含 1/N）
        grad_kernel = (tiles * dz[..., None]).sum(axis=(0, 1)).reshape(2, 2)
        grad_bias = dz.sum()
        return loss, grad_kernel, grad_bias

    def save(self, path: str) -> None:
        """权重落盘 = 真产线的 checkpoint 导出。"""
        obj = {"version": self.version, "kernel": self.kernel.tolist(), "bias": float(self.bias)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "DetectorCore":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return cls(version=obj["version"], kernel=np.array(obj["kernel"]), bias=obj["bias"])


# ─────────────────────────── 2. 计算图雏形（模拟 ONNX） ───────────────────────────

def _op_conv2x2(x: np.ndarray, weights: dict) -> np.ndarray:
    k = np.asarray(weights["kernel"])
    h, w = x.shape
    tiles = np.stack([
        x[0:h-1, 0:w-1], x[0:h-1, 1:w],
        x[1:h,   0:w-1], x[1:h,   1:w],
    ], axis=-1)
    z = tiles @ k.reshape(-1) + weights["bias"]
    return 1.0 / (1.0 + np.exp(-z))


def _op_threshold(x: np.ndarray, weights: dict) -> np.ndarray:
    """元素级：>= thr → 1，否则 0。ONNX 里是 Greater + Where。"""
    return (x >= weights["thr"]).astype(np.int8)


GRAPH_OPS = {"conv2x2_sigmoid": _op_conv2x2, "threshold": _op_threshold}


def build_graph(core: DetectorCore, thr: float = 0.5) -> list:
    """把检测器表示成"图"——ONNX 的玩具版。

    真实世界：torch.onnx.export(model, dummy, ...) 产出等价的东西（protobuf 格式）。
    注意权重是图的常量输入（ONNX 叫 initializer）——E2 实验的 Q3 考点。
    """
    return [
        {"op": "conv2x2_sigmoid",
         "inputs": ["image"],
         "weights": {"kernel": core.kernel.tolist(), "bias": core.bias},
         "out": "heatmap"},
        {"op": "threshold",
         "inputs": ["heatmap"],
         "weights": {"thr": thr},
         "out": "binary"},
    ]


def graph_forward(ops: list, feeds: dict) -> dict:
    """按拓扑序执行计算图。这就是所有推理引擎（ORT/TRT）的主循环骨架：
    for 节点 in 图: 取输入 → 查 kernel 表 → 写输出。
    """
    values = dict(feeds)
    for node in ops:
        fn = GRAPH_OPS[node["op"]]
        args = [values[name] for name in node["inputs"]]
        values[node["out"]] = fn(*args, weights=node["weights"])
    return values


# ─────────────────────────── 3. 后处理：decode + NMS ───────────────────────────

@dataclass
class Box:
    x: int; y: int; score: float


def decode(heatmap: np.ndarray, thr: float = 0.5) -> list:
    """热力图 → 候选框列表。玩具版：每个超阈值位置一个 2×2 框。"""
    ys, xs = np.where(heatmap >= thr)
    return [Box(int(x), int(y), float(heatmap[y, x])) for y, x in zip(ys, xs)]


def iou(a: Box, b: Box, size: int = 2) -> float:
    """两个 size×size 框的 IoU（交并比）。"""
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + size, a.y + size
    bx0, by0, bx1, by1 = b.x, b.y, b.x + size, b.y + size
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = size * size * 2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list, iou_thr: float = 0.5) -> list:
    """非极大值抑制：按分数降序，贪心保留，压掉和已保留框重叠过高的。

    和图形学遮挡剔除同构：按"重要性"排序、贪心通过、抑制邻居。
    真实 2D 检测里 NMS 在后处理段，CPU 上跑，量大了会成瓶颈——
    这也是为什么有 NMS-free 的 DETR 一派。
    """
    boxes = sorted(boxes, key=lambda b: -b.score)
    keep = []
    for b in boxes:
        if all(iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep


# ─────────────────────────── 自测 ───────────────────────────

def self_test() -> None:
    rng = np.random.default_rng(3)
    core = DetectorCore()
    # 数值梯度验证 backward 写对了（产线参与训练前也该这么验）
    x = rng.normal(0, 1, (9, 9))
    target = (rng.random((8, 8)) > 0.7).astype(float)
    loss, gk, gb = core.loss_grad(x, target)
    eps = 1e-6
    for idx in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        kp = core.kernel.copy(); kp[idx] += eps
        km = core.kernel.copy(); km[idx] -= eps
        cp = DetectorCore(kernel=kp, bias=core.bias)
        cm = DetectorCore(kernel=km, bias=core.bias)
        num = (cp.loss_grad(x, target)[0] - cm.loss_grad(x, target)[0]) / (2 * eps)
        assert abs(num - gk[idx]) < 1e-4, (idx, num, gk[idx])
    # 图执行 = 手写 forward
    hm_direct = core.forward(x)
    hm_graph = graph_forward(build_graph(core), {"image": x})["heatmap"]
    assert np.allclose(hm_direct, hm_graph)
    # NMS 行为：iou_thr=0.1 时 (0,0) 与 (1,1) 的 1/7 重叠被抑制，(6,6) 保留
    boxes = [Box(0, 0, 0.9), Box(1, 1, 0.8), Box(6, 6, 0.7)]
    kept = nms(boxes, 0.1)
    assert len(kept) == 2 and kept[0].score == 0.9 and kept[1].score == 0.7, kept
    print("[D2] 算子内核自测通过：数值梯度 / 图执行一致 / NMS")


if __name__ == "__main__":
    self_test()
