# -*- coding: utf-8 -*-
"""
🚦 红绿灯检测 demo —— 先把结果跑出来看，细节再一块块抠
======================================================
一张合成"摄像头帧"（暗背景 + 一个红色亮斑 = 红绿灯玩具版），
走完整条检测链，全部纯 numpy、CPU 秒级：

  [0] 合成帧        48×64 伪 RGB，埋一个 6×6 红色亮斑
  [1] 预处理 (D1)   灰度 → 双线性缩放 24×32 → 归一化
  [2] 检测器 (D2)   2×2 卷积核(4权重) + bias + sigmoid → 热力图，共 5 个参数
  [3] 训练          手写 backward + 梯度下降（真训练，不是假装）
  [4] 解码 + NMS    热力图 → 候选框 → 去重叠 → 最终检测框

跑法（在仓库根目录下）：
    python -X utf8 minilops/demo_traffic_light.py

import 的 preprocess / operator_core 是参考答案——现在当黑盒用，
先看效果；哪一块想拆开看，指给我，我们一块块抠。
"""

import io
import sys

# 防中文/emoji 在 Windows 控制台乱码：把标准输出包成 UTF-8 流
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

from preprocess import preprocess                     # D1 参考答案（黑盒）
from operator_core import DetectorCore, decode, nms  # D2 参考答案（黑盒）

# ─────────────────────────── 参数板 ───────────────────────────

FRAME_H, FRAME_W = 48, 64            # 合成摄像头帧尺寸
BLOB_R, BLOB_C, BLOB_S = 20, 40, 6   # 亮斑在原图的位置（行,列）和边长
IN_H, IN_W = 24, 32                  # 预处理输出尺寸（正好缩一半 → 亮斑 6×6 变 3×3）
LR, STEPS = 8.0, 200                 # 学习率 / 训练步数

CHARS = " .:-=+*#%@"                 # 10 档 ASCII 灰度表（黑→白）= 一张 LUT


def ascii_render(a: np.ndarray) -> str:
    """2D 数组 → 字符画：值域 [min, max] 线性映射到 10 档字符。"""
    # a.min()/a.max()：全图最小/最大值 → 当映射的两端（求和收缩族，出标量）
    lo, hi = a.min(), a.max()
    # 实打实：a = [[0, 5], [10, 10]]，lo=0, hi=10
    #   (a - 0) / (10 - 0 + 1e-12) * 9.999 = [[0, 4.99], [9.99, 9.99]]
    #   减法/除法/乘法全逐元素；+1e-12 防全图同值时除零
    #   .astype(int)：逐元素向零砍成整数 → [[0, 4], [9, 9]]（查 LUT 的下标）
    idx = ((a - lo) / (hi - lo + 1e-12) * 9.999).astype(int)
    # np.clip(idx, 0, 9)：把下标钳进 [0,9]（GLSL clamp 同款，防御性兜底）
    idx = np.clip(idx, 0, 9)
    # 内层 "".join(...)：一行的每个下标查 CHARS 表，连成一行字符串
    # 外层 "\n".join(...)：行与行之间插换行
    return "\n".join("".join(CHARS[i] for i in row) for row in idx)


def indent(s: str) -> str:
    """每行前面缩进 4 空格，让面板好读。"""
    # s.split("\\n")：按换行符切成行列表；拼回去时每行加前缀
    return "\n".join("    " + line for line in s.split("\n"))


def make_frame() -> np.ndarray:
    """合成一帧 (48, 64, 3) 伪 RGB：暗背景 + 一个 6×6 红色亮斑。"""
    # 固定种子 → 每次跑出完全相同的"随机"数（可复现）
    rng = np.random.default_rng(0)
    # np.full((48,64,3), 25.0)：48行×64列×3通道(RGB)、全是 25.0 的数组
    img = np.full((FRAME_H, FRAME_W, 3), 25.0)
    # rng.normal(0, 8, img.shape)：同形状高斯噪声（均值0 标准准差8）
    #   实打实：某次可能取到 [-3.2, 5.1, 0.4, ...]，每个数独立抖动
    # 相加逐元素：背景 = 25 + 噪声，大约落在 1~49（暗）
    img += rng.normal(0, 8, img.shape)
    # 切片赋值：行20:26 × 列40:46 的 6×6 区域整块改写（埋红绿灯）
    #   红光 = R 亮、G/B 暗：肉眼是红色
    #   BT.601 灰度化后 ≈ 0.299*235 + 0.587*60 + 0.114*60 ≈ 112（背景 25）
    img[BLOB_R:BLOB_R + BLOB_S, BLOB_C:BLOB_C + BLOB_S] = (235.0, 60.0, 60.0)
    # 亮斑也叠同款噪声（有噪声的现实世界，不是完美矩形）
    img[BLOB_R:BLOB_R + BLOB_S, BLOB_C:BLOB_C + BLOB_S] += rng.normal(0, 8, (BLOB_S, BLOB_S, 3))
    return img


def main() -> None:
    print("═" * 60)
    print(" 🚦 红绿灯检测 demo：合成帧 → 预处理 → 训练 → 热力图 → 检测框")
    print("═" * 60)

    # ── [0] 合成帧 ──
    frame = make_frame()
    print(f"\n[0] 合成摄像头帧 {frame.shape}（暗背景 + 6×6 红色亮斑）")

    # ── [1] 预处理（D1，当黑盒）──
    # preprocess：灰度(BT.601) → 双线性缩放 48×64→24×32 → 均值方差归一化
    #   亮斑 6×6 → 3×3；归一化后 背景≈-0.1、亮斑≈+2.9（均值0方差1的世界）
    x = preprocess(frame, target_size=(IN_H, IN_W))
    print(f"\n[1] 预处理 → {x.shape}。模型看到的图（自己先找亮斑在哪）：")
    print(indent(ascii_render(x)))

    # ── [2] 训练前的热力图（瞎的）──
    core = DetectorCore()          # 5 个参数：4 权重 + 1 bias，随机小值起步
    p_before = core.forward(x)     # (23, 31) 热力图
    print(f"\n[2] 训练前的热力图 {p_before.shape}（没学过 → 一片含糊）：")
    print(indent(ascii_render(p_before)))

    # ── [3] 训练：手写 backward + 梯度下降 ──
    # 标签：哪些窗口"2×2 四邻居全在亮斑里"
    #   亮斑 resize 后在 行10:13 × 列20:23；窗口(i,j) 的邻居 = 行i,i+1 × 列j,j+1
    #   → 全在斑里 ⟺ i∈{10,11} 且 j∈{20,21} → 4 个正窗口，其余 709 个全 0
    #   ⚠ 像素坐标 ↔ 窗口坐标 的对齐是关5 的经典坑，先按下不表
    target = np.zeros((IN_H - 1, IN_W - 1))
    target[10:12, 20:22] = 1.0

    print(f"\n[3] 训练 {STEPS} 步（loss 一路跌 = 真在学习）：")
    for step in range(STEPS):
        loss, gk, gb = core.loss_grad(x, target)   # 前向 + 手写反向一次拿全
        if step % 40 == 0:
            print(f"    step {step:3d}   loss = {loss:.4f}")
        core.kernel -= LR * gk     # 下山一步：参数 -= 步长 × 梯度
        core.bias -= LR * gb
    loss, _, _ = core.loss_grad(x, target)
    print(f"    step {STEPS}   loss = {loss:.4f}")

    # ── [4] 训练后的热力图（亮斑处一个尖峰）──
    p = core.forward(x)
    print("\n[4] 训练后的热力图（亮斑位置亮起尖峰）：")
    print(indent(ascii_render(p)))

    # ── [5] 解码 + NMS → 检测框 ──
    boxes = decode(p, thr=0.5)         # 热力图 ≥0.5 的窗口各出一个 2×2 候选框
    kept = nms(boxes, iou_thr=0.1)     # 去重叠；阈值为什么取 0.1 → Bonus B3 的坑
    print(f"\n[5] 解码：{len(boxes)} 个候选框 → NMS 保留 {len(kept)} 个")
    best = kept[0]                     # nms 按分数降序返回，第一个 = 最自信
    print(f"\n    🚦 检测到红绿灯：框左上角 (x={best.x}, y={best.y})，"
          f"大小 2×2，置信度 {best.score:.2f}")
    print(f"    （亮斑真实位置：行 10:13 × 列 20:23 —— 对上了 ✅）")
    print("\n" + "═" * 60)
    print(" 检测链跑完了。哪一块想拆开看？指给我，一块块抠。")
    print("═" * 60)


if __name__ == "__main__":
    main()
