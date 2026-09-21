# -*- coding: utf-8 -*-
"""
D1 · 预处理算子 —— 产线算子的输入适配层
=========================================
真实产线对应物：图片进模型前的 resize / 归一化 / 颜色空间转换。
2D 预标注算子的预处理通常是：letterbox resize → /255 → mean/std 归一化 → HWC→CHW。

这里做玩具版：伪 RGB 图（numpy 数组）→ 缩放 → grayscale → 均值归一化。

为什么预处理值得单独一个文件？
因为它是产线算子最容易和训练侧"悄悄不一致"的地方——
训练时 normalize 用了 (0.5, 0.5, 0.5)，部署时忘了，精度掉一半都不报错。
所以真实工程里，预处理会写成独立、可测试、有单测的单元（就像这里）。
"""

import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 → (H, W) float64。BT.601 亮度加权，和前端 CSS filter 同族。"""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"期望 (H, W, 3)，得到 {img.shape} —— 算子的输入契约检查")
    weights = np.array([0.299, 0.587, 0.114])  # BT.601，YUV 的 Y
    return img.astype(np.float64) @ weights


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(H, W) → (out_h, out_w)，双线性插值。
    不调 cv2/PIL：亲手写一遍就知道 resize 本身是个卷积类运算，
    也顺便体会 numpy 的 grid 生成 + 花式索引（gather）能力。
    """
    if img.ndim != 2:
        raise ValueError(f"期望 2D 图，得到 {img.shape}")
    h, w = img.shape
    if h == out_h and w == out_w:
        return img.copy()
    # 源坐标网格：目标像素中心映射回源图（align_corners=False 语义）
    row_ratio = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    col_ratio = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    row = np.clip(row_ratio, 0, h - 1)
    col = np.clip(col_ratio, 0, w - 1)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    # 四个邻居 + 线性权重 —— bilinear = 两次 lerp，图形学老朋友
    wr = (row - r0)[:, None]  # (out_h, 1) 广播到列
    wc = (col - c0)[None, :]  # (1, out_w)
    top = img[np.ix_(r0, c0)] * (1 - wc) + img[np.ix_(r0, c1)] * wc
    bot = img[np.ix_(r1, c0)] * (1 - wc) + img[np.ix_(r1, c1)] * wc
    return top * (1 - wr) + bot * wr


def normalize(img: np.ndarray) -> np.ndarray:
    """(H, W) → (H, W)，均值方差归一化到 0 附近。模拟训练侧的 mean/std。"""
    mu = img.mean()
    std = img.std() + 1e-8
    return (img - mu) / std


def preprocess(img: np.ndarray, target_size=(16, 16)) -> np.ndarray:
    """完整预处理管道（这就是算子的"前处理"段）。"""
    g = to_grayscale(img)
    g = resize_bilinear(g, *target_size)
    return normalize(g)


def self_test() -> None:
    """单测：算子必须有的东西。跑 run_all 时会自动执行。"""
    img = np.random.default_rng(0).integers(0, 256, (37, 41, 3)).astype(np.uint8)
    out = preprocess(img, (16, 16))
    assert out.shape == (16, 16), out.shape
    assert abs(out.mean()) < 1e-9, "归一化后均值应≈0"
    assert 0.2 < out.std() < 5, "std 异常"

    # 契约测试：坏输入必须在入口处炸，不能带病进模型
    try:
        to_grayscale(np.zeros((5, 5)))
        raise AssertionError("2D 输入应该报错")
    except ValueError:
        pass

    # resize 幂等性：同尺寸返回原值
    g = np.arange(12).reshape(3, 4).astype(float)
    assert np.allclose(resize_bilinear(g, 3, 4), g)
    print("[D1] 预处理算子自测通过：契约检查 / 归一化 / resize 幂等")


if __name__ == "__main__":
    self_test()
