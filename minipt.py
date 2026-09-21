# -*- coding: utf-8 -*-
"""
MiniPT —— 迷你 Pointcept:亲手搓一条点云语义分割全链路
=====================================================
合成 LiDAR(S1) → GridSample(S2) → Collect(S3) → attention(S5)
→ 六积木组装(S6) → 训练五步心跳(S4) → mIoU 评估(S9)

这是你的练习场:核心函数留了 TODO(共 ~80 行),架子、模型、自检全部备好。
一关一关来:实现 → 自检 → 过。卡住 15 分钟就来问 Claude(问法模板见手册 §5)。

用法(都在仓库根目录,注意 -X utf8):
    python -X utf8 minipt.py --check 0    # 环境自检(torch/numpy)
    python -X utf8 minipt.py --check 1    # 第 1 关:合成一帧"LiDAR 扫描"
    python -X utf8 minipt.py --check 2    # 第 2 关:GridSample(字典版)
    python -X utf8 minipt.py --check 3    # 第 3 关:Collect(进 torch 世界)
    python -X utf8 minipt.py --check 4    # 第 4 关:softmax 与 attention
    python -X utf8 minipt.py --check 5    # 第 5 关:训练一步(五步心跳)
    python -X utf8 minipt.py --check 6    # 第 6 关:mIoU
    python -X utf8 minipt.py --check all  # 从 0 连跑到第一个没过的关
    python -X utf8 minipt.py --train      # 全链路训练 200 步 → 毕业(mIoU ≥ 0.85)
    python -X utf8 minipt.py --train --steps 400   # 没到 0.85 就加步数
    python -X utf8 minipt.py --compare    # 实验:不做 GridSample 有多惨(亲手感受"为什么")

配套手册:minipt_手册.md —— 每关的"为什么/提示/对照真身/常见坑"都在里面。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # 占位:没装 torch 时文件仍可 import(第 1/2 关不需要 torch)。
    # 真正用到时(第 3 关起/定义模型)才报错,提示装法。
    class _Stub:
        def __getattr__(self, name):
            if name in ("Module", "ModuleList", "Linear", "LayerNorm",
                        "Sequential", "Parameter", "Tensor"):
                return _Stub
            raise ImportError("torch 未安装:pip install torch --index-url https://download.pytorch.org/whl/cpu")
    torch = nn = F = _Stub()

# ─────────────────────────── 全局配置 ───────────────────────────
CFG = dict(
    seed=42,
    # 第 1 关:场景(一个 20m×20m 的迷你世界)
    ground_size=20.0, n_ground=15000,   # 地面:z≈0 的薄板,类 0
    car_center=(4.0, 3.0, 0.0), n_car=3000,   # 车:高斯 blob,类 1
    pole_x=-6.0, pole_y=-2.0, n_pole=1200,    # 电线杆:竖直圆柱,类 2
    # 第 2 关:GridSample
    grid_size=0.6,
    # 第 3~5 关:模型与训练
    dim=32, n_blocks=2, lr=0.01, steps=200,
    # 类别
    n_classes=3, class_names=["ground", "car", "pole"],
)


# ══════════════════════════════════════════════════════════════
# 第 0 关:环境(已写好)
# ══════════════════════════════════════════════════════════════
def check_env():
    print("numpy", np.__version__)
    if not HAS_TORCH:
        print("❌ 还没装 torch。安装(CPU 版,约 200MB):")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False
    print("torch", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
    print("   (本机没 CUDA 也没关系,MiniPT 全程 CPU 可跑)")
    return True


# ══════════════════════════════════════════════════════════════
# 第 1 关:合成一帧"LiDAR 扫描"(TODO)
# ══════════════════════════════════════════════════════════════
# 为什么:Pointcept 一切的入口是"一帧点云 = 标准字典"(S1/S3)。
# 我们造一个迷你世界:地面(类0) + 一辆车(类1) + 一根电线杆(类2)。
# 真身对照:pointcept/datasets/semantic_kitti.py —— 读 .bin 得 [N,4],
# 翻译成同一个标准字典。字典里每个数组必须同长度、同顺序(第 N 个 segment
# 描述第 N 个 coord)—— 这个约定贯穿全仓库。

def make_scene(rng: np.random.Generator) -> dict:
    """造一帧合成点云,返回标准字典 {"coord": (N,3), "segment": (N,)}。

    已写好:地面(z≈0 的均匀薄板)。
    你来写:车 + 电线杆 + 拼接。自检会验证的规格:

      车   n_car 个点,xy 中心在 CFG["car_center"] 的前两维,
           高斯 sigma=(1, 1, 0.4),z 不许穿到地下(clip 到 ≥ 0)
      杆   n_pole 个点,立在 (CFG["pole_x"], CFG["pole_y"]),
           半径 0.08、高 4m 的竖直圆柱(角度 uniform(0, 2π),z uniform(0, 4))
      拼接 coord (N,3) 与 segment (N,),三段依次是 0/1/2

    提示(全是 step1_touch_pointcloud.py 写过的):
      rng.normal(中心, sigma, (n,3)) / rng.uniform(lo, hi, n)
      np.stack([...], axis=1) / np.concatenate([...], axis=0)
      np.clip(car[:, 2], 0, None)   # A8/A4 的 clip,穿地部分抬回地面
    """
    n_ground = CFG["n_ground"]
    half = CFG["ground_size"] / 2
    ground = np.stack([
        rng.uniform(-half, half, n_ground),
        rng.uniform(-half, half, n_ground),
        rng.uniform(-0.05, 0.05, n_ground),
    ], axis=1)
    segment_ground = np.zeros(n_ground, dtype=np.int64)

    # ── TODO 1a:车(n_car, 3)── z 记得 clip 到 ≥ 0 ──────────────

    # ── TODO 1b:电线杆(n_pole, 3)竖直圆柱 ─────────────────────

    # ── TODO 1c:拼三个 coord、三段 segment,返回标准字典 ────────

    raise NotImplementedError("第 1 关:车 + 杆 + 拼接(见函数内 TODO,手册 §2.1)")


def check1():
    rng = np.random.default_rng(42)
    d = make_scene(rng)
    assert isinstance(d, dict), "返回值应是 dict"
    c, s = d["coord"], d["segment"]
    n = CFG["n_ground"] + CFG["n_car"] + CFG["n_pole"]
    assert c.shape == (n, 3) and s.shape == (n,), \
        f"形状:coord {c.shape} / segment {s.shape},期望 ({n},3) / ({n},)"
    assert np.isfinite(c).all(), "坐标里出现 nan/inf"
    assert sorted(np.unique(s).tolist()) == [0, 1, 2], "三类都要出现"
    assert (s == 0).sum() == CFG["n_ground"], "地面点数与配置不一致"
    assert (s == 1).sum() == CFG["n_car"], "车点数与配置不一致"
    assert (s == 2).sum() == CFG["n_pole"], "杆点数与配置不一致"
    car = c[s == 1]
    pole = c[s == 2]
    cx, cy, _ = CFG["car_center"]
    assert car[:, 2].min() >= 0.0, "车穿地了(z < 0):对 z 做 clip"
    assert (np.abs(car[:, 0] - cx) < 5).all() and (np.abs(car[:, 1] - cy) < 5).all(), \
        "车的 xy 应在中心 ±5m 内(sigma=1 的 5σ)"
    assert pole[:, 2].max() > 3.0, "杆不够高:最高 z 应 > 3m(高 4m 的圆柱)"
    assert pole[:, 2].min() >= -1e-9, "杆穿地了(z < 0)"
    assert (np.abs(pole[:, 0] - CFG["pole_x"]) < 0.5).all(), "杆的 x 应贴着杆位 ±0.5m"
    assert (np.abs(pole[:, 1] - CFG["pole_y"]) < 0.5).all(), "杆的 y 应贴着杆位 ±0.5m"
    d2 = make_scene(np.random.default_rng(42))
    assert np.allclose(d["coord"], d2["coord"]), "同种子应产出同一帧(伪随机确定性,s1q2)"
    print("✅ 第 1 关通过 —— 你会造标准字典了(真身:semantic_kitti.py 读 .bin 干的就是这件事)")


# ══════════════════════════════════════════════════════════════
# 第 2 关:GridSample,字典版(TODO)
# ══════════════════════════════════════════════════════════════
# 为什么:19k 个点做 attention 是 N² 的灾难;体素化后 ~1000 token,
# CPU 也能秒训(S2/S5)。你在 step2_gridsample.py 实验三写过纯坐标版,
# 这里升级成"字典版"——新难点:coord 抽走哪些点,segment 必须跟走哪些点。
# 对不齐 = 训练全毁,这是真实工程事故第一名(自检专门有一项抓这个)。
# 真身对照:pointcept/datasets/transform.py:840(GridSample)。
# 真身还多两套(scale 细网格 / inverse 铺回,bonus 关 B2 会玩),核心五步与你的完全一致。

def grid_sample(data: dict, grid_size: float, rng: np.random.Generator) -> dict:
    """每个体素留一个随机代表点。输入输出都是标准字典(处理 coord/segment)。

    五步食谱(S2 讲解;先 -=min 再打包,和 step2 实验三同款):
      ① gc = np.floor(coord / grid_size).astype(np.int64)   # 格坐标=格身份(floor 左下角锚点)
      ② gc -= gc.min(axis=0)                                 # 平移到非负(打包前必须,负数会撞车)
      ③ key = gc[:,0]*10**10 + gc[:,1]*10**5 + gc[:,2]       # 10^5 进制打包成一个大数(RGB24 同构)
      ④ idx_sort = np.argsort(key, kind="stable")             # 下标表:谁排第几(s2q2)
         _, count = np.unique(key, return_counts=True)        # 每格点数
         start = np.insert(count, 0, 0).cumsum()[:-1]         # 每格起点(前缀和,s2q3)
      ⑤ pick = start + rng.integers(0, count.max(), len(count)) % count   # 组内随机折回(s2q4)
         idx = idx_sort[pick]
    返回 {"coord": coord[idx], "segment": segment[idx]}

    注意:随机数用传入的 rng(比 step2 的 np.random.randint 全局版升级:
    随机也纳入种子管理,同种子可复现 —— 自检会验证这一点)。
    """
    coord = np.asarray(data["coord"])
    segment = np.asarray(data["segment"])
    # ── TODO 2:按食谱实现,每步一行,共 ~6 行 ───────────────────

    raise NotImplementedError("第 2 关:五步食谱(见函数内注释,手册 §2.2)")


def check2():
    # 自带小场景,不依赖第 1 关:8 个点、5 个体素(grid=1.0)
    coord = np.array([
        [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1],
        [1.2, 0.9, 0.1], [0.2, 0.2, 1.1], [5.0, 5.0, 5.0], [5.4, 5.1, 5.2],
    ])
    segment = np.array([0, 0, 1, 2, 1, 0, 2, 2])
    data = {"coord": coord, "segment": segment}
    out = grid_sample(data, 1.0, np.random.default_rng(7))
    oc, os_ = np.asarray(out["coord"]), np.asarray(out["segment"])
    assert len(oc) == 5, f"8 点 5 格,应剩 5 点,得到 {len(oc)}(检查 unique/起点那两行)"
    # 每个输出点必须真的来自输入,且 coord 和 segment 成对不串线
    pairs_in = {(tuple(coord[i]), int(segment[i])) for i in range(len(coord))}
    for i in range(len(oc)):
        assert (tuple(oc[i]), int(os_[i])) in pairs_in, \
            f"第 {i} 个输出点 coord 与 segment 对不上(串线了!idx 要同时用在两个数组上)"
    # 输出点落在互不相同的体素里(一格一代表 = 去重语义)
    keys = {tuple(np.floor(oc[i] / 1.0).astype(int)) for i in range(len(oc))}
    assert len(keys) == 5, "输出的格坐标应互不相同"
    # 同种子两次结果一致(随机走传入的 rng 才能做到)
    o1 = grid_sample(data, 1.0, np.random.default_rng(7))
    o2 = grid_sample(data, 1.0, np.random.default_rng(7))
    assert np.allclose(o1["coord"], o2["coord"]) and (o1["segment"] == o2["segment"]).all(), \
        "同种子应可复现:随机数请用传入的 rng(rng.integers),不要用全局 np.random"
    print("✅ 第 2 关通过 —— 字典版 GridSample:coord/segment 同步走(真身:transform.py:840)")


# ══════════════════════════════════════════════════════════════
# 第 3 关:Collect —— numpy 字典 → torch 张量(TODO,很短但概念重)
# ══════════════════════════════════════════════════════════════
# 为什么:transform 管道是 numpy 世界,模型是 torch 世界,Collect 是
# 两岸的桥(S3):① feat 按顺序拼起来(coord 本身也进 feat!)② 转 tensor。
# 真身:pointcept/datasets/transform.py:54 —— feat_keys=["coord","strength"]
# 拼出 (N,4),所以 config 里 in_channels=4(s3q2/s9q5)。我们这里用
# ["coord"] → (N,3),所以模型 in_channels=3。
# 注意:真身 Collect 还有一件大事 —— 生成 offset(batch 分段账本)。
# 但 offset 是 batch>1 才需要的东西,MiniPT 全程 batch=1,所以挪到
# bonus 关 B1 领略,主线不挡路。

def collect(data: dict) -> dict:
    """字典版 Collect:numpy → torch。

    返回 {"feat": tensor(N,3) float32, "segment": tensor(N,) int64}
    feat 就是 np.stack([data["coord"]], axis=1) 再转 torch.float32。

    提示:torch.from_numpy(x).float() / .long()
    """
    # ── TODO 3:两三行 ─────────────────────────────────────────

    raise NotImplementedError("第 3 关:拼 feat + 转 tensor(手册 §2.3)")


def check3():
    d = {"coord": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
         "segment": np.array([0, 2])}
    out = collect(d)
    assert HAS_TORCH and torch.is_tensor(out["feat"]) and torch.is_tensor(out["segment"]), \
        "输出的值应是 torch tensor(torch.from_numpy)"
    assert out["feat"].shape == (2, 3) and out["feat"].dtype == torch.float32, \
        f"feat 期望 (N,3) float32,得到 {tuple(out['feat'].shape)} {out['feat'].dtype}"
    assert out["segment"].dtype == torch.int64, \
        f"segment 期望 int64(交叉熵的 target 必须 long),得到 {out['segment'].dtype}"
    assert torch.allclose(out["feat"], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    print("✅ 第 3 关通过 —— Collect:两岸之桥(真身:transform.py:54,feat_keys 拼 feat → in_channels)")


# ══════════════════════════════════════════════════════════════
# 第 4 关:softmax 与 attention 的心脏(TODO)
# ══════════════════════════════════════════════════════════════
# 为什么:attention = "token 间开会":每个 token 看一圈邻居、
# 按相似度打分、softmax 变权重、加权混货物(S5)。
# 这里做最小版:score 已经替你算好(q·k),你来完成"打分 → 权重 → 混货"
# 的后两步 —— 正是 GroupedVectorAttention(v2m2_base.py:120~127)
# 的 :122(softmax)和 :127(加权求和)两行。
# 真身还会减 max 做数值稳定(softmax 溢出问题),这里也要求做:
# e = exp(score - score.max(-1)) —— 每行减自己的最大值再 exp。

def softmax(x: "torch.Tensor") -> "torch.Tensor":
    """数值稳定版 softmax,最后一维。两三行:
      e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
      return e / e.sum(dim=-1, keepdim=True)
    """
    # ── TODO 4a ───────────────────────────────────────────────

    raise NotImplementedError("第 4a 关:softmax(手册 §2.4)")


def attend(score: "torch.Tensor", value: "torch.Tensor") -> "torch.Tensor":
    """score (N, S), value (N, S, C) → 输出 (N, C)。

    每个查询 token 有 S 个候选邻居:score[n, s] 是第 s 个的分数,
    value[n, s, :] 是第 s 个的货物。输出 = 全部货物的加权和。

    两步:① w = softmax(score)      权重和为 1(N, S)
          ② out = (w.unsqueeze(-1) * value).sum(dim=1)   消掉 S 维
    对照真身 v2m2_base.py:127:(attn.unsqueeze(-1) * v).sum(2) —— 一模一样,
    只是维数从 (B,H,N,K) 降到了 (N,S)。
    """
    # ── TODO 4b ───────────────────────────────────────────────

    raise NotImplementedError("第 4b 关:加权混货(手册 §2.4)")


def check4():
    s = torch.softmax(torch.tensor([[2.0, 1.0, 0.1]]), dim=-1)[0]
    mine = softmax(torch.tensor([[2.0, 1.0, 0.1]]))[0]
    assert torch.allclose(mine, s, atol=1e-6), f"softmax 数值不对:{mine} vs {s}"
    big = softmax(torch.tensor([[1000.0, 1000.0]]))
    assert torch.isfinite(big).all() and torch.allclose(big.sum(-1), torch.tensor(1.0)), \
        "大数溢出了:先减 max 再 exp"
    score = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    value = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 10.0]]])
    out = attend(score, value)
    # 第 0 行:权重 (e/(e+1), 1/(e+1)) ≈ (0.731, 0.269) → 0.731*(1,0) + 0.269*(0,1)
    w = torch.exp(torch.tensor(1.0)) / (torch.exp(torch.tensor(1.0)) + 1)
    assert torch.allclose(out[0], torch.tensor([w.item(), 1 - w.item()]), atol=1e-5), \
        f"attend 第 0 行期望 ≈ (0.731, 0.269),得到 {out[0].tolist()}"
    # 第 1 行:权重偏向第 1 个 → 主要取 (0,10)
    assert out[1, 1] > out[1, 0] * 5, "权重没生效:分数高的邻居应占主导"
    print("✅ 第 4 关通过 —— softmax + 加权混货 = attention 的心脏(真身:v2m2_base.py:122/127)")


# ══════════════════════════════════════════════════════════════
# 第 5 关:训练一步 —— 五步心跳(TODO,全项目画龙点睛)
# ══════════════════════════════════════════════════════════════
# 为什么:torch 里所有训练循环都是同一心跳(S4):
#   zero_grad → forward → loss → backward → step
# 你在 S4 学过"为什么":PyTorch 梯度是累加语义,不清零会累加出错误梯度。
# 这里亲手写一次,并亲眼看到 backward 之后每个权重都拿到了 .grad。
# 真身:engines/train.py train_step(还多三样工程加固:梯度累积/AMP/clip,
# 你的迷你版不需要 —— 但 s8q4 考的就是这三样)。

def train_step(model, feat, segment, optimizer) -> float:
    """跑一个训练步,返回 loss 的数值(float)。

    五步(每步一行,顺序就是心跳顺序):
      optimizer.zero_grad()
      logits = model(feat)                    # forward(注意:写 model(x) 不写 model.forward(x),走 __call__ 才带钩子)
      loss = F.cross_entropy(logits, segment) # loss(和真身一样:CE 对 19 类,我们对 3 类)
      loss.backward()                         # backward(引擎自动算梯度,不需要你手推导数)
      optimizer.step()                        # 下山一步
    最后 return loss.item()(item():单元素 tensor → python float,日志用)
    """
    # ── TODO 5:五步心跳 ───────────────────────────────────────

    raise NotImplementedError("第 5 关:五步心跳(手册 §2.5)")


def check5():
    torch.manual_seed(0)
    model = MiniPT(CFG["dim"], CFG["n_classes"], n_blocks=1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    feat = torch.randn(200, 3)
    segment = torch.randint(0, 3, (200,))
    # 第一次:手写版
    loss = train_step(model, feat, segment, opt)
    assert isinstance(loss, float) and np.isfinite(loss), "应返回 float(loss.item())"
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert grads_ok, "backward 没给权重发梯度(.grad 是 None?)—— 心跳里 backward 在 step 之前"
    # 对照版:同模型同数据,五步顺序标准版跑两次,看趋势
    torch.manual_seed(0)
    m2 = MiniPT(CFG["dim"], CFG["n_classes"], n_blocks=1)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=0.01)
    l_ref = train_step_reference(m2, feat, segment, opt2)
    torch.manual_seed(0)
    m3 = MiniPT(CFG["dim"], CFG["n_classes"], n_blocks=1)
    opt3 = torch.optim.AdamW(m3.parameters(), lr=0.01)
    l_before = F.cross_entropy(m3(feat), segment).item()
    train_step(m3, feat, segment, opt3)
    l_after = F.cross_entropy(m3(feat), segment).item()
    assert l_after < l_before, \
        f"训练一步后 loss 应下降:{l_before:.3f} → {l_after:.3f}(step 漏了?或 zero_grad/顺序错了?)"
    assert abs(l_ref - loss) < 1e-4, "与参考实现数值不一致(心跳顺序或 loss 写法不同?)"
    print("✅ 第 5 关通过 —— 五步心跳闭环,backward 发梯度、step 真下山(真身:train.py train_step)")


def train_step_reference(model, feat, segment, optimizer) -> float:
    optimizer.zero_grad()
    loss = F.cross_entropy(model(feat), segment)
    loss.backward()
    optimizer.step()
    return loss.item()


# ══════════════════════════════════════════════════════════════
# 第 6 关:mIoU —— 分割模型的真考分(TODO,三行)
# ══════════════════════════════════════════════════════════════
# 为什么:分类看准确率,分割看 mIoU(S9):每类算"交并比"再平均。
# 直觉:ground 类大、pole 类小,准确率会被 ground 淹没(97% 看着很高,
# pole 可能全错);IoU 每类平等,骗不了人 —— 这正是 config 里
# class_weight 从 1.3(vegetation)到 10.2(motorcyclist)的动机(s9q3)。
# 真身:pointcept/engines/evaluate.py(SemSegEvaluator,挂在 after_epoch)。

def miou(pred: "torch.Tensor", target: "torch.Tensor", n_classes: int) -> float:
    """pred/target: (N,) 的类别下标。返回各类 IoU 的平均(忽略从未出现的类)。

    三行食谱:
      intersect = torch.bincount(...)  # 各类命中数
      用混淆矩阵最稳:conf = torch.zeros(n_classes, n_classes);
        conf[target, pred] += 1     # 行=真值 列=预测(花式索引+= 是 scatter 加法)
      每类 iou_i = conf[i,i] / (conf[i].sum() + conf[:,i].sum() - conf[i,i])
      出现过的类(分母>0)取平均
    """
    # ── TODO 6:混淆矩阵 → 每类 IoU → 平均 ────────────────────

    raise NotImplementedError("第 6 关:mIoU(手册 §2.6)")


def check6():
    t = torch.tensor([0, 0, 0, 1, 1, 2])
    p = torch.tensor([0, 0, 1, 1, 1, 0])
    # 混淆矩阵(行=真值):类0 命中2/错1;类1 命中2;类2 全错
    # iou0 = 2/(3+2-2) = 0.667;iou1 = 2/(2+2-2) = 1.0;iou2 = 0/(1+1-0) = 0
    # mIoU = (0.667+1+0)/3 ≈ 0.556
    got = miou(p, t, 3)
    expected = (2 / 3 + 1.0 + 0.0) / 3   # ≈ 0.5556
    assert abs(got - expected) < 1e-6, f"mIoU 期望 ≈ {expected:.4f},得到 {got}"
    assert 0.0 <= got <= 1.0
    # 全对 → 1.0
    assert abs(miou(t, t, 3) - 1.0) < 1e-6, "全对应得 1.0"
    print("✅ 第 6 关通过 —— mIoU:每类平等的真考分(真身:evaluate.py SemSegEvaluator)")


# ══════════════════════════════════════════════════════════════
# 毕业项目:MiniPT 网络 + 训练器(全部已写好 —— 你来读)
# ══════════════════════════════════════════════════════════════
# 通关六个 TODO 之后,这个文件里剩下的部分不用写,但要"读得懂":
# 它是六积木(S6)的真人组装秀 —— 你已经认识每一块积木了。
#
#   embed  :Linear(3 → dim)     每点独立升维(PointNet 式,无邻域概念)
#   blocks :× n_blocks 个 TransformerBlock
#       ├ 注意力:每点找 k 近邻 → score → attend(S5 你写的那个函数!)
#       ├ 残差  :x = x + drop(attn(x))     (s6q1:梯度高速路)
#       ├ MLP   :Linear→GELU→Linear       (s6q2:万能逼近)
#       └ LayerNorm:稳压器                 (s6q4)
#   head   :Linear(dim → 3)     输出每类 logits
#
# 每个类后都标了它对应复习站哪一章 —— 读完顺手回站点做几道题巩固。

class NeighborhoodAttention(nn.Module):
    """kNN 邻域 attention —— PTv2 的"16-NN 邻域"最小复刻。

    真身 v2m2_base.py:48(GroupedVectorAttention)有 Q/K/V 三投影和分组;
    最小版直接用坐标差当 score 的原始信号,保留的是结构:
    邻居集合 → 打分 → softmax → 混货,以及 mask 处理(不足 k 个的边距)。
    """

    def __init__(self, dim, k=8):
        super().__init__()
        self.k = k
        self.proj_q = nn.Linear(dim, dim)   # Q 投影
        self.proj_v = nn.Linear(dim, dim)   # V 投影(货物)
        self.score_mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, 1))
        # 打分头:吃 [查询点坐标, 邻居坐标] 的差(简化版相对位置编码 ——
        # PTv2 三改动之三)

    def forward(self, x, coord):
        N = x.shape[0]
        k = min(self.k, N)
        # ① kNN:每点的 k 近邻(真身在 CUDA kernel 里做,这里用 torch 的算子)
        d = torch.cdist(coord, coord)                     # (N, N) 两两距离
        idx = d.topk(k, largest=False).indices            # (N, k) 最近 k 个(含自己)
        nb_x = x[idx]                                     # (N, k, C) 邻居货物 gather!
        nb_c = coord[idx]                                 # (N, k, 3)
        # ② 打分:坐标差过一个小 MLP(相对位置编码的信号)
        rel = torch.cat([coord.unsqueeze(1).expand_as(nb_c), nb_c], dim=-1)  # (N, k, 6)
        score = self.score_mlp(rel).squeeze(-1)           # (N, k)
        # ③ 加权混货 —— 第 4 关写的 attend!
        v = self.proj_v(nb_x)                             # (N, k, C)
        out = attend(score, v)                            # (N, C)
        return self.proj_q(x) + out                       # 残差:自己 + 邻居信息


class TransformerBlock(nn.Module):
    """六积木真人版(S6 讲解的 Block.forward 的迷你同款)。

    真身:v2m2_base.py:132 Block.forward —— 结构完全同构,只少了
    分组向量化与 serialization(串行化排序)两个工程加速件。
    """

    def __init__(self, dim, k=8, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)                    # 稳压器
        self.attn = NeighborhoodAttention(dim, k)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(                         # token 内变换
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.drop_path = DropPath(drop_path)              # 随机深度

    def forward(self, x, coord):
        x = x + self.drop_path(self.attn(self.norm1(x), coord))   # 残差①
        x = x + self.mlp(self.norm2(x))                           # 残差②
        return x


class DropPath(nn.Module):
    """DropPath(s6q3):训练时整块跳过(p 概率),推理时全开。

    真身:models/utils.py DropPath —— 隐式 ensemble;
    注意它与 model.train()/model.eval() 的联动(开关由模式控制)。
    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, device=x.device) >= self.p
        return x * keep / (1 - self.p)   # 保持期望值不变(和 Dropout 同款缩放)


class MiniPT(nn.Module):
    """全网络:embed → blocks → head。"""

    def __init__(self, dim=32, n_classes=3, n_blocks=2):
        super().__init__()
        self.embed = nn.Linear(3, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, n_classes)

    def forward(self, feat):
        coord = feat[:, :3]                # 前 3 维就是坐标(我们的 feat 只有 coord)
        x = self.embed(feat)
        for b in self.blocks:
            x = b(x, coord)
        return self.head(x)                # (N, 3) 每类 logits


def make_batch(rng):
    """一帧新场景(数据增强也在这:随机平移 + 旋转 —— S3 增强链迷你版)。

    真身 config:99-123:Rotate/Scale/Flip/Jitter 一整链;我们用一个
    随机旋转矩阵 + 平移浓缩表达。点云增强的本质:坐标变、标签不变
    (旋转车还是车)—— 语义分割增强免费的根源。
    """
    d = make_scene(rng)
    ang = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang),  np.cos(ang), 0],
                  [0, 0, 1]])
    d["coord"] = d["coord"] @ R.T + rng.uniform(-1, 1, 3)
    return grid_sample(d, CFG["grid_size"], rng)


def train(args):
    if not all_checks_fast():     # 通关才能毕业训练(报哪关没过)
        return
    print(f"MiniPT 毕业训练:{args.steps} 步,合成场景 + 随机旋转增强 + GridSample {CFG['grid_size']}")
    rng = np.random.default_rng(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    model = MiniPT(CFG["dim"], CFG["n_classes"], n_blocks=CFG["n_blocks"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        d = make_batch(rng)                 # 新场景(带增强)→ GridSample
        batch = collect(d)                  # 进 torch 世界
        loss = train_step(model, batch["feat"], batch["segment"], opt)   # 你的五步心跳
        if step == 1 or step % 20 == 0:
            with torch.no_grad():
                d = make_batch(np.random.default_rng(999))   # 固定验证场景
                batch = collect(grid_sample(d, CFG["grid_size"], np.random.default_rng(999)))
                pred = model(batch["feat"]).argmax(-1)
                m = miou(pred, batch["segment"], CFG["n_classes"]).item()   # 你的 mIoU
            print(f"  step {step:>4} | loss {loss:.3f} | 验证 mIoU {m:.3f}"
                  + ("  🎓" if m >= 0.85 else ""))
            if m >= 0.85:
                print(f"\n🎓 毕业了!{step} 步,mIoU {m:.3f}(≥ 0.85)。"
                      f"你亲手搓通了:合成 LiDAR → GridSample → Collect → attention"
                      f" → 六积木 → 五步心跳 → mIoU —— Pointcept 主链路一条全通。")
                print("   接下来:① 试试 --compare 亲手感受 GridSample 的价值;")
                print("           ② 手册的 bonus 关(offset / inverse 铺回);")
                print("           ③ 把训练曲线和复习站 S4/S5/S6 的题对上号。")
                return
    print(f"\n到 {args.steps} 步 mIoU 还没到 0.85(当前 {m:.3f})。"
          f"加步数试试:python -X utf8 minipt.py --train --steps 400")


def run_compare():
    """对照实验:同样的训练步数,不做 GridSample(全 19k 点 kNN attention)。"""
    if not all_checks_fast():
        return
    print("对照实验:GridSample 0.6 (≈1000 token) vs 不采样 (≈19k token)")
    print("预测:不采样版每步慢一个数量级以上(N² 的 cdist/topk),还不一定更准 ——")
    print("      采样丢的是冗余(地面上一格 60 个点本来就说的同一件事)\n")

    def run(tag, use_gs):
        rng = np.random.default_rng(CFG["seed"])
        torch.manual_seed(CFG["seed"])
        model = MiniPT(CFG["dim"], CFG["n_classes"], n_blocks=CFG["n_blocks"])
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
        t0 = time.time()
        for step in range(1, 41):
            d = make_scene(rng)
            if use_gs:
                d = grid_sample(d, CFG["grid_size"], rng)
            else:
                d = {"coord": d["coord"], "segment": d["segment"]}
            batch = collect(d)
            loss = train_step(model, batch["feat"], batch["segment"], opt)
            if step % 20 == 0:
                print(f"  [{tag}] step {step:>3} | {time.time()-t0:6.1f}s | loss {loss:.3f}")
        print(f"  [{tag}] 40 步耗时 {time.time()-t0:.1f}s\n")

    run("GridSample", True)
    run("不采样  ", False)
    print("看到了吗 —— 这就是 config 里 GridSample 排在模型前面的原因:")
    print("token 数 = 计算量的平方根号(N² attention);采样 = 用 1/20 的点换 400 倍速度,")
    print("而语义信息几乎不丢(每格的代表点就够说清'这格是地面')。")


# ══════════════════════════════════════════════════════════════
# 自检系统 + 命令行(已写好)
# ══════════════════════════════════════════════════════════════

CHECKS = [
    ("第 1 关 · 合成一帧 LiDAR", check1),
    ("第 2 关 · GridSample(字典版)", check2),
    ("第 3 关 · Collect(进 torch)", check3),
    ("第 4 关 · softmax 与 attention", check4),
    ("第 5 关 · 训练一步(五步心跳)", check5),
    ("第 6 关 · mIoU", check6),
]


def all_checks_fast() -> bool:
    """静默跑全部自检(毕业训练前把关),失败时打印是哪关。"""
    try:
        if not check_env():
            return False
        for name, fn in CHECKS:
            try:
                fn()
            except NotImplementedError as e:
                print(f"❌ {name} 还没实现:{e}")
                return False
            except AssertionError as e:
                print(f"❌ {name} 自检失败:{e}")
                return False
    except SystemExit:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="MiniPT —— 迷你 Pointcept 练习场")
    ap.add_argument("--check", default="all", help="0/1/2/3/4/5/6/all")
    ap.add_argument("--train", action="store_true", help="毕业训练(需先通关)")
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    ap.add_argument("--compare", action="store_true", help="GridSample 对照实验")
    args = ap.parse_args()

    if args.train:
        train(args)
        return
    if args.compare:
        run_compare()
        return

    if args.check == "all":
        if not check_env():
            return
        for name, fn in CHECKS:
            try:
                fn()
            except NotImplementedError as e:
                print(f"\n⏸ {name} 还没实现 —— 从这里开始:\n    {e}")
                print(f"   运行 python -X utf8 minipt.py --check N 只跑这一关")
                return
            except AssertionError as e:
                print(f"\n❌ {name} 自检失败:{e}")
                return
        print("\n🎉 全部 6 关通过!可以毕业训练了:python -X utf8 minipt.py --train")
        return

    n = int(args.check)
    if n == 0:
        check_env()
    else:
        try:
            CHECKS[n][1]()
        except NotImplementedError as e:
            print(f"⏸ 还没实现:{e}")
        except AssertionError as e:
            print(f"❌ 自检失败:{e}\n   对照手册 §2.{n} 的提示改一改再跑")


if __name__ == "__main__":
    main()
