# -*- coding: utf-8 -*-
"""
D3 · 工作流引擎 + 模拟产线 —— "集成到工作流"本体
=========================================
真实产线对应物：标注平台的 workflow/pipeline 编排系统。
一帧图片进产线 → 过一串算子 → 产出预标注 → 置信度不够的进人工队列 → 修正后回流。

三个关键机制（都从这里体验）：

1. REGISTRY 注册表 —— 和 Pointcept 的 registry.py 同构思想：
   算子类先注册（"我叫 sobel_edge"），编排时用字符串引用（"sobel_edge"）。
   为什么不直接传函数？因为配置文件里只能写字符串——
   config 驱动 = 注册表存在的全部理由（Vue 的 app.component + <component :is> 同款）。

2. 算子协议 —— 每个算子是一个类，实现 process(data) -> data：
   输入输出都是同一个 dict（"数据信封"），算子只读/写自己的字段。
   真实产线同理：算子间靠 schema 解耦，谁也不 import 谁。

3. 审查点（review）—— 预标注的灵魂：
   算子给出的结果带置信度，够高的直接通过（省人工），
   不够的打回人工修（保证质量）。采纳率 = 通过/总数 = 产线的核心 KPI。
"""

from preprocess import preprocess
from operator_core import DetectorCore, decode, nms


# ─────────────────────────── 算子协议与注册表 ───────────────────────────

class Operator:
    """算子基类：真实产线的算子基类会有更多——版本、资源声明、超时。"""
    name = "base"

    def process(self, data: dict) -> dict:
        raise NotImplementedError


REGISTRY: dict = {}   # name -> 算子类


def register(cls):
    """装饰器注册——Pointcept 的 @MODELS.register_module() 玩具版。"""
    if cls.name in REGISTRY:
        raise ValueError(f"算子重名: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


@register
class GrayscaleNormOp(Operator):
    """D1 的管道包成算子：信封里写 normalized 字段。"""
    name = "grayscale_norm"

    def __init__(self, target_size=(16, 16)):
        self.target_size = target_size

    def process(self, data):
        data["normalized"] = preprocess(data["image"], self.target_size)
        return data


@register
class DetectorOp(Operator):
    """D2 的内核包成算子：预标注框 + 每框置信度。

    version 记进输出——"这帧的框是哪个版本的模型标的"必须可追溯，
    回流训练时才知道数据是谁产的（产线算子的硬要求）。
    """
    name = "detector"

    def __init__(self, core: DetectorCore, conf_thr=0.5, iou_thr=0.5):
        self.core = core
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    def process(self, data):
        hm = self.core.forward(data["normalized"])
        boxes = decode(hm, self.conf_thr)
        data["pred_boxes"] = nms(boxes, self.iou_thr)
        data["pred_version"] = self.core.version
        return data


@register
class RoiFilterOp(Operator):
    """下游业务算子：只关心 ROI（画面中心区域）内的目标。
    体现"工作流"的意义：算子能自由组合，业务逻辑不用改上游。
    """
    name = "roi_filter"

    def __init__(self, margin=2):
        self.margin = margin

    def process(self, data):
        h, w = data["normalized"].shape
        boxes = data["pred_boxes"]
        kept = []
        for b in boxes:
            cx, cy = b.x + 1, b.y + 1   # 2×2 框中心
            if self.margin <= cx < w - self.margin and self.margin <= cy < h - self.margin:
                kept.append(b)
        data["roi_boxes"] = kept
        return data


@register
class ReviewGateOp(Operator):
    """审查门：置信度不够 → 打回人工。这就是"预标注"里"人"出现的位置。

    真实产线：高置信度自动通过（省标注员的时间），低置信度进人工队列。
    阈值是业务调参点：太高→采纳率低省不了人力；太低→错标混进回流毒化训练。
    """
    name = "review_gate"

    def __init__(self, min_conf=0.75):
        self.min_conf = min_conf

    def process(self, data):
        boxes = data["pred_boxes"]
        best = max((b.score for b in boxes), default=0.0)
        if boxes and best >= self.min_conf:
            data["review"] = "auto_pass"
        else:
            data["review"] = "needs_human"
        return data


# ─────────────────────────── 工作流引擎 ───────────────────────────

def run_workflow(tasks: list, op_specs: list) -> list:
    """跑批：tasks 是数据信封列表，op_specs 是算子描述（名字 + kwargs）。

    注意 op_specs 完全是 JSON 能表达的形态——真实产线的 pipeline 就是一份
    配置文件（yaml/json），引擎按配置实例化算子再串联。
    这就是"算子注册 + config 驱动 + 引擎执行"三件套的全貌。
    """
    # 惰性实例化：同一配置跑多批，算子只实例化一次（真实产线还要考虑
    # 模型只加载一次、常驻内存——D4 服务化会体现这点）
    ops = [REGISTRY[spec["name"]](**spec.get("kwargs", {})) for spec in op_specs]
    results = []
    for task in tasks:
        data = dict(task)   # 信封复制，避免算子间互相污染
        try:
            for op in ops:
                data = op.process(data)
            data["status"] = "done"
        except Exception as e:   # 失败处理：真实产线必须接住，不能让一帧挂整条线
            data["status"] = f"error: {type(e).__name__}: {e}"
        results.append(data)
    return results


# ─────────────────────────── 模拟数据与产线 ───────────────────────────

def make_synthetic_frame(rng, size=24, n_targets=3):
    """造一帧"路口图"：暗底 + 若干亮块（= 玩具版红绿灯）。

    亮块是 3×3 高斯亮斑——2×2 卷积核能学会"响应四邻居都亮的位置"。
    返回 (image, target_heatmap)：target 就是训练/回流要用的真值。
    """
    img = rng.random((size, size, 3)) * 40                # 暗噪声底
    target = np.zeros((size - 1, size - 1))
    for _ in range(n_targets):
        cx, cy = rng.integers(4, size - 4, 2)
        blob = rng.random() * 60 + 150                    # 亮块亮度
        img[cy-1:cy+2, cx-1:cx+2] += blob * 0.35
        img[cy, cx] += blob * 0.4
        target[cy-1, cx-1] = 1.0                          # 2×2 感受野标签
    return np.clip(img, 0, 255).astype(np.uint8), target


import numpy as np  # 放这里防止截断后单独 import 丢 np


def build_pipeline(core, min_conf=0.75):
    """产线配置——真实世界这就是一份 yaml。"""
    return [
        {"name": "grayscale_norm", "kwargs": {"target_size": (23, 23)}},
        {"name": "detector", "kwargs": {"core": core, "conf_thr": 0.5, "iou_thr": 0.5}},
        {"name": "roi_filter", "kwargs": {"margin": 2}},
        {"name": "review_gate", "kwargs": {"min_conf": min_conf}},
    ]


def run_demo(n_frames=24, seed=42):
    """模拟产线跑批：看多少帧自动通过、多少帧打回人工。"""
    rng = np.random.default_rng(seed)
    core = DetectorCore()   # 未训练的模型——预标注质量差，就该大量打回
    tasks = []
    for i in range(n_frames):
        img, target = make_synthetic_frame(rng, size=23)  # 23→预处理后 23→卷积后 22，与标签 22 对齐
        tasks.append({"task_id": f"frame_{i:03d}", "image": img, "target": target})
    results = run_workflow(tasks, build_pipeline(core))

    auto = sum(1 for r in results if r["review"] == "auto_pass")
    human = sum(1 for r in results if r["review"] == "needs_human")
    err = sum(1 for r in results if r["status"].startswith("error"))
    print(f"[D3] 产线跑批 {n_frames} 帧：自动通过 {auto} / 打回人工 {human} / 出错 {err}")
    print(f"     （未训练模型 → 预标注质量差 → 大量打回，这就是回流的原料）")
    return results, tasks


def self_test() -> None:
    results, _ = run_demo(n_frames=12, seed=1)
    assert all(r["status"] == "done" for r in results)
    assert all("review" in r for r in results)
    print("[D3] 工作流引擎自测通过：注册表 / 信封传递 / 审查门")


if __name__ == "__main__":
    self_test()
