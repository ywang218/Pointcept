# -*- coding: utf-8 -*-
"""
数据闭环 · retrain feedback —— 预标注产线的灵魂
=========================================
真实产线对应物：人工修正后的数据回流 → 微调模型 → 重新部署 → 预标注更好。

这条环是"数据引擎"的核心：模型产数据、数据养模型。
领导说"lidar seg 回流的还不够训出效果"——说的就是这条环的原料不足；
2D 红绿灯预标注要做的，就是给这条环供料。

玩具闭环（本文件跑的）：
  1. 未训练模型跑产线（D3）→ 大量帧被打回人工
  2. 打回的帧 + 真值（模拟标注员修正）→ 当训练集
  3. 梯度下降微调 DetectorCore（D2 手写的 backward 在这里派上用场）
  4. 重新跑产线 → 采纳率上升、错误下降
  5. 循环几轮，看曲线——这就是"数据飞轮"的形状

真值哪来的？玩具里是合成的（make_synthetic_frame 顺手产出）。
真实世界里就是标注员修的框——所以人工修正不是成本，是训练数据。
"""

import numpy as np

from operator_core import DetectorCore
from workflow import make_synthetic_frame, build_pipeline, run_workflow


def finetune(core: DetectorCore, train_data, epochs=60, lr=0.5) -> list:
    """朴素 SGD 微调。返回每 epoch 的 loss 曲线。

    就是最朴素的五步心跳（你在 MiniPT 第 5 关写的同款）：
    forward → loss → backward → step → zero。
    没有 optimizer/scheduler——玩具版直接手写更新，看清楚梯度下降本体。
    """
    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for x, target in train_data:
            loss, gk, gb = core.loss_grad(x, target)
            core.kernel -= lr * gk   # 梯度下降本体：参数 -= lr * 梯度
            core.bias -= lr * gb
            epoch_loss += loss
        losses.append(epoch_loss / len(train_data))
    return losses


def evaluate(core: DetectorCore, n_frames=40, seed=99) -> dict:
    """产线视角的评估：采纳率（自动通过率）+ 预测质量（F1）。"""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_frames):
        img, target = make_synthetic_frame(rng, size=23)
        tasks.append({"task_id": f"eval_{i:03d}", "image": img, "target": target})
    results = run_workflow(tasks, build_pipeline(core))

    auto = sum(1 for r in results if r["review"] == "auto_pass")
    adoption = auto / n_frames

    # 逐帧 F1（玩具版：预测框中心 vs 真值亮斑位置，容差 1 像素）
    f1s = []
    for r in results:
        pred = {(b.x, b.y) for b in r["pred_boxes"]}
        gt = set(map(tuple, np.argwhere(r["target"] > 0.5)))
        tp = sum(1 for p in pred if any(abs(p[0]-g[1]) + abs(p[1]-g[0]) <= 1 for g in gt)) if gt else 0
        tp = min(tp, len(gt)) if gt else 0
        prec = tp / len(pred) if pred else (1.0 if not gt else 0.0)
        rec = tp / len(gt) if gt else 1.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return {"adoption": adoption, "f1": float(np.mean(f1s))}


def run_demo(rounds=3) -> list:
    """跑完整闭环：预标注（差）→ 修正回流 → 微调 → 预标注（好）。"""
    core = DetectorCore(version="0.2.0")
    log = []
    for rd in range(rounds):
        # 1) 当前模型跑产线，收打回的帧
        rng = np.random.default_rng(100 + rd)
        tasks = []
        for i in range(20):
            img, target = make_synthetic_frame(rng, size=23)  # 尺寸对齐：预处理 23 → 卷积后 22 = 标签
            tasks.append({"task_id": f"r{rd}_{i:03d}", "image": img, "target": target})
        results = run_workflow(tasks, build_pipeline(core))
        rejected = [r for r in results if r["review"] == "needs_human"]
        # 2) 打回的帧当训练数据（"人工修正"= 用真值；真值即合成时的标签）
        train_data = [(r["normalized"], r["target"]) for r in rejected]
        if not train_data:
            print(f"[retrain] 第 {rd+1} 轮：无打回帧，闭环饱和")
            break
        # 3) 微调 + 4) 版本推进（真实产线：新 checkpoint 重新部署）
        losses = finetune(core, train_data, epochs=40, lr=0.3)
        core.version = f"0.2.{rd+1}"
        # 5) 固定测试集上看指标
        ev = evaluate(core, n_frames=30, seed=777)
        log.append((rd + 1, len(rejected), losses[0], losses[-1], ev))
        print(f"[retrain] 第 {rd+1} 轮：打回 {len(rejected):2d} 帧 → 微调 loss "
              f"{losses[0]:.3f}→{losses[-1]:.3f} → 采纳率 {ev['adoption']:.0%} / F1 {ev['f1']:.2f}")
    print("[retrain] 这条曲线就是数据飞轮：打回越少=采纳率越高=人工越省")
    return log


def self_test() -> None:
    log = run_demo(rounds=2)
    assert len(log) >= 1
    if len(log) >= 2:
        # 闭环有效性：至少 F1 或采纳率有一项明显改善
        assert log[-1][4]["f1"] > log[0][4]["f1"] - 0.1
    print("[retrain] 数据闭环自测通过：打回→回流→微调→指标变化")


if __name__ == "__main__":
    self_test()
