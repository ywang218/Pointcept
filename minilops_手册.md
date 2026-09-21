# MiniLOps 学习手册 —— 7 关闯关版

> 配套：`minilops/` 里的五个文件（preprocess / operator_core / workflow / retrain_feedback / server）是**参考答案**，学习期间别通读，过关一关翻一关。
> 你的作业写在 `minilops/my_d1.py` ~ `my_d7.py`：**只准依赖 numpy（关7 另加 flask）和前一关的自己写的文件，不许 import 参考答案**。
> 每关流程：读本关讲解 → 写代码 → 自检全绿 → 翻参考答案对照 diff → 喊 Claude 点评 → 进下一关。
> 节奏建议：一晚 1~2 关，总共约 5 小时。跑自己的文件记得 `python -X utf8 minilops/my_d1.py`。

---

## 全景地图：为什么是这个顺序

```
原始图 ─→ [关1]预处理成模型输入 ─→ [关2]模型出热力图 ─→ [关3]热力图变框(NMS)
       ─→ [关4]框进可编排的流水线 ─→ [关5]流水线加质量门(打回人工)
       ─→ [关6]打回的回流养模型(飞轮) ─→ [关7]整体包成服务
```

每一关只在前一关"已经能跑的东西"上加一层——你永远不会一次性面对空白页。
这条链也是真实建产线的先后：先能算一张图 → 再能批处理 → 再有质量门 → 再有闭环 → 再服务化。

---

## 热身（10 分钟，现在就做）

跑 `python -X utf8 minilops/run_all.py`，**只看输出，别读代码**。
把每一行输出归位到上面地图：`[D1]` 对应关1、`[D2]` 对应关2/3、`[D3]` 对应关4/5、`[retrain]` 两行对应关6。
看到"采纳率 83%→87%"那两行时记住这个感觉：**7 关之后，那两个数字会出自你自己的手**。

---

## 关卡 1：预处理算子（30-40 分钟）

**为什么先做它**：预处理是产线算子和训练侧最容易"悄悄不一致"的地方——训练时归一化用 (0.5,0.5,0.5)、部署时忘了，精度掉一半都不报错。所以它必须写成**独立、有契约检查、有单测的单元**。这一关就是练"算子的输入适配层怎么写"。

**你要写** `my_d1.py`，三个函数，契约：

```python
to_grayscale(img)        # (H, W, 3) uint8 → (H, W) float64，BT.601 亮度加权
resize_bilinear(img, out_h, out_w)   # (H, W) → (out_h, out_w)，双线性插值
normalize(img)           # (H, W) → (H, W)，均值 0 方差 1（std 加 1e-8 防除零）
```

**自检**（写进 `if __name__ == "__main__"`，全绿才算过）：

```python
img = np.random.default_rng(0).integers(0, 256, (37, 41, 3)).astype(np.uint8)
out = normalize(resize_bilinear(to_grayscale(img), 16, 16))
assert out.shape == (16, 16)
assert abs(out.mean()) < 1e-9          # 归一化后均值≈0
assert 0.2 < out.std() < 5
# 契约：2D 输入必须在入口炸 ValueError，不能带病进模型
try: to_grayscale(np.zeros((5, 5))); raise AssertionError("应报错")
except ValueError: pass
# resize 幂等：同尺寸返回原值
g = np.arange(12).reshape(3, 4).astype(float)
assert np.allclose(resize_bilinear(g, 3, 4), g)
```

**提示**：
- grayscale 一个矩阵乘就够：`img.astype(np.float64) @ [0.299, 0.587, 0.114]`。这三个权重 = 人眼对红绿蓝的亮度敏感度，和 CSS `filter: grayscale(1)` 是同一张表——你的前端老朋友。
- resize 的坐标映射（align_corners=False 语义）：目标像素 `(i)` 中心映回源图 `(i + 0.5) * H / out_h - 0.5`，然后 `floor + clip` 到边界，取四个邻居做**两次 lerp**（先行后列或先列后行都行）。bilinear 就是 lerp 套 lerp——图形学基本功。
- `np.ix_(r, c)` 一次取出四邻居子矩阵，比双循环快两个量级。

**常见坑**：忘记 `astype(float)` 导致整数除法/整除截断；clip 边界忘了（源坐标越界）；权重顺序写错变成 `(3,) @ img`。

**参考答案**：`minilops/preprocess.py`（过关前别翻）。

---

## 关卡 2：检测器内核——forward + 手写 backward（40-60 分钟，本套最重的一关）

**为什么做它**：这是"模型"本体。手写 backward 不是虐待——关6 的回流微调要**真训练**（不是假装），全靠这一关的梯度。你会在 30 行里走完"前向 → 和标签比 → 链式法则"的最小闭环，也就是 MiniPT 第 5 关那套五步心跳的去框架版。

**模型定义**（玩具版红绿灯检测器）：单通道 2×2 卷积（4 个权重 + 1 个 bias）→ sigmoid → 热力图。热力图高分位置 = 有目标。

**你要写** `my_d2.py`，两个函数，契约：

```python
forward(kernel, bias, x)      # kernel:(2,2) x:(H,W) → (H-1, W-1) sigmoid 热力图
loss_grad(kernel, bias, x, target)   # target:(H-1,W-1) 的 0/1 标签
                              # → (loss, grad_kernel:(2,2), grad_bias: float)
```

loss 用 BCE（把"每个位置是不是目标"当二分类，对全图求平均）。

**自检——数值梯度验证**（手写 backward 的黄金标准，写对了才配往下走）：

```python
# 对 4 个权重位置逐一验证：解析梯度 vs 数值梯度
eps = 1e-6
# num = (L(kernel + eps*e_ij) - L(kernel - eps*e_ij)) / (2*eps)
# assert |num - grad_kernel[i,j]| < 1e-4
```

**提示**：
- 2×2 卷积不用循环：把四个邻居 stack 成 `(H-1, W-1, 4)` 的 tiles（im2col 的迷你版），再 `tiles @ kernel.reshape(-1) + bias`。
- BCE 对 z 的梯度是 `(p - target) / N`（N = 像素总数，因为 loss 取了 mean）；有了 dz，对 kernel 的梯度 = tiles 加权求和，对 bias 的梯度 = dz 求和。链式法则自己推一遍再写——推不出来再看这行。
- sigmoid 导数用不着显式算（BCE+sigmoid 复合后 dz 恰好是 p−target），想想为什么——这是 BCE 配 sigmoid 的经典美事。

**常见坑**：梯度里忘了除 N（loss 是 mean）；tiles 顺序和 kernel reshape 顺序不对应（forward/backward 必须同一套摆法）。

**参考答案**：`minilops/operator_core.py` 的 `DetectorCore`。

---

## 关卡 3：后处理——decode + IoU + NMS（30 分钟）

**为什么做它**：热力图不是框。检测算子的输出契约是"框列表"，decode（热力图→候选）+ NMS（去重叠）就是后处理段。NMS 是 2D 检测部署的必考题，而且它和图形学**遮挡剔除**同构：按重要性排序、贪心保留、抑制邻居。

**你要写** `my_d3.py`，契约：

```python
decode(heatmap, thr)     # → 框列表，每框 (x, y, score)；(y, x) = np.where(heatmap >= thr)
iou(a, b, size=2)        # 两个 size×size 框的交并比
nms(boxes, iou_thr)      # 按分数降序，贪心保留，压掉与已保留框 IoU≥thr 的
```

**自检**（这道题是我搭服务时真实踩的坑，送你）：

```python
# 手算：2×2 框错位 1 像素，IoU 是多少？先纸面算，再用代码验证
assert abs(iou(Box(0,0,.9), Box(1,1,.8)) - 1/7) < 1e-6   # 交 1，并 4+4-1=7

boxes = [(0, 0, 0.9), (1, 1, 0.8), (6, 6, 0.7)]
kept = nms(boxes, iou_thr=0.1)   # (0,0) 与 (1,1) 的 1/7 重叠被压掉
assert len(kept) == 2 and kept[0][2] == 0.9 and kept[1][2] == 0.7
```

**提示**：IoU = 交集面积 / 并集面积；交集 = `max(0, min(ax1,bx1)-max(ax0,bx0))` 两个方向各一次；排序用 `sorted(boxes, key=..., reverse=True)`。

**常见坑**：交集忘了 clip 到 0（错位远的框交出负面积）；先压再排（必须先排序再贪心）。

**参考答案**：`minilops/operator_core.py` 的 decode/iou/nms。

---

## 关卡 4：工作流引擎——注册表 + 编排（30-40 分钟）

**为什么做它**：领导那句"集成到工作流"的本体。三个机制：**注册表**（算子类先注册，编排时用字符串引用——和 Pointcept 的 registry 同构，Vue 的 `app.component` + `<component :is>` 同款）；**信封协议**（算子间传同一个 dict，谁也不 import 谁，靠字段解耦）；**config 驱动**（编排描述是纯 JSON 能表达的形态）。

**你要写** `my_d4.py`，契约：

```python
class Operator:                # 基类：name + process(data)->data
def register(cls):             # 装饰器：写进 REGISTRY，重名要炸 ValueError
REGISTRY: dict                 # name → 类
def run_workflow(tasks, op_specs)   # tasks: 信封列表；op_specs: [{"name":..., "kwargs":{...}}]
                                 # → 结果列表；单个信封炸不许挂整批（status 记 error）
```

再写两个最小算子验证引擎：`GrayscaleNormOp`（包你关1的管道）和 `DetectorOp`（包你关2 的 forward + 关3 的 decode/nms）。

**自检**：
```python
# ① 重名注册炸
# ② 信封缺 image 字段 → 该信封 status 记 error，同批其他信封照常 done
# ③ 两个算子串联：第一个写 normalized 字段，第二个能读到
```

**提示**：惰性实例化——`run_workflow` 开头把 op_specs 实例化一次（真实产线的对应物：模型只加载一次）；信封要 `dict(task)` 复制，避免算子间互相污染。

**常见坑**：实例化放进帧循环里（每帧重建"模型"）；except 写得太窄接不住所有异常。

**参考答案**：`minilops/workflow.py` 上半部分。

---

## 关卡 5：审查门 + 模拟产线（20-30 分钟）

**为什么做它**：预标注的灵魂在这一关出现——"人"。审查门规则：预标注置信度够高直接通过（省人工），不够打回人工（保质量）。**采纳率 = 自动通过 / 总数**，是产线的核心 KPI；阈值是业务调参点：太高省不了人力，太低错标混进回流毒化训练。

**你要写** `my_d5.py`：
```python
make_synthetic_frame(rng, size=23)   # 暗噪声底 + 若干亮斑；返回 (image, target)
ReviewGateOp                          # best score ≥ min_conf → auto_pass，否则 needs_human
```
然后组装产线（norm → detector → review_gate）跑 12 帧。

**自检**：未训练的随机小 kernel → **12 帧全打回**（预标注质量差就该这样——打回的就是回流的原料）。

**⚠️ 天然的坑（我真实踩过，别绕过去）**：帧尺寸用 23，不是 24。链路是 `23×23 图 → 预处理后 23×23 → 2×2 卷积后 22×22`，所以 target 必须是 22×22 才对得上——用 24 会死在关6 的 loss 广播错误上。**这是"预处理/卷积输出尺寸对齐"的经典坑的微缩版**：真实 2D 检测里 letterbox 后要不要再除以 stride、小数怎么取整，全是同一族问题。

**提示**：亮斑造法——3×3 区域加亮、中心最亮，2×2 卷积才能学会"响应四邻居都亮的位置"。

**参考答案**：`minilops/workflow.py` 的 make_synthetic_frame / ReviewGateOp / run_demo。

---

## 关卡 6：数据闭环——回流微调（40 分钟，灵魂关）

**为什么做它**：领导说"lidar seg 回流还不够训出效果"——说的就是这条环的原料不足。这一关你亲手转飞轮：**预标注（差）→ 打回 + 真值（= 人工修正）→ 回流微调 → 预标注（好）**。用你关2 手写的 backward 做真梯度下降，五步心跳的无框架版：forward → loss → backward → `参数 -= lr * 梯度` → 下一轮。

**你要写** `my_d6.py`：
```python
finetune(kernel, bias, train_data, epochs, lr)   # 朴素 SGD，返回 loss 曲线列表
evaluate(kernel, bias, n_frames, seed)           # → {"adoption": ..., "f1": ...}
run_demo(rounds=3)                               # 跑闭环，打印每轮：打回几帧/loss 变化/采纳率/F1
```

**自检**：第 1 轮打回约 20 帧 → 微调 loss 显著下降 → 采纳率明显上升、打回数变少。**如果打回数不降或 F1 不动，先查 lr（太大震荡太小不动，0.3 附近合适）再查关2 的梯度**。

**提示**：训练数据 = 打回帧的 `(normalized, target)`；"人工修正"在玩具里 = 直接用合成时的真值（真实世界里这就是标注员修的框——**人工修正不是成本，是训练数据**，这句话是整条产线的价值观）；F1 用预测框中心 vs 真值亮斑位置、容差 1 像素算 TP。

**参考答案**：`minilops/retrain_feedback.py`。

---

## 关卡 7：算子服务化（30 分钟）

**为什么做它**："接入工作流"的最后一公里——平台不 import 你的代码，只按 schema 调你。三个产线灵魂机制全在这一关：**模型只加载一次常驻进程**（不是每请求 load）、**可观测性**（/health 探活 + /stats 延迟指标）、**输入契约校验**（坏请求 4xx 带原因，一帧坏数据不许挂整条线）。

**你要写** `my_d7.py`（flask）：
```python
POST /detect   {"image": H×W×3 数组} → {"boxes": [...], "model_version": ..., "n": ...}
GET  /health   → {"ok": true, "version": ..., "uptime_s": ...}
GET  /stats    → {"n_req": ..., "n_err": ..., "latency_ms": {"avg":..., "p95":...}}
```

**自检**（三条 curl + 预期）：
```bash
python -X utf8 minilops/my_d7.py &
curl http://127.0.0.1:8791/health                 # 200 ok
curl -X POST .../detect -d '{"bad": 1}'           # 400 + error 说明缺 image
curl -X POST .../detect -d '{"image": [[1,2],[3,4]]}'   # 400 + shape 错误说明
curl http://127.0.0.1:8791/stats                  # 有延迟数字
```

**提示**：`np.asarray(payload["image"], dtype=np.uint8)` 然后查 `ndim/shape`；延迟用 `time.perf_counter()` 差值；注意用**另一个端口**（8790 可能还被参考答案的服务占着）；练好的 kernel 直接当全局 CORE（加载一次）。

**参考答案**：`minilops/server.py`。

---

## Bonus 破坏性实验（通关后，最快建立直觉）

- **B1 铁律验证**：把 `CORE` 从全局搬进 `detect()` 里（每请求重建）→ 打 10 发 curl 对比 /stats 的延迟涨幅。亲手体会"模型只加载一次"为什么是铁律。
- **B2 阈值毒化**：审查门 `min_conf` 调到 0.1 → 采纳率虚高但 F1 掉——错标混进回流毒化训练的样子。这就是"业务调参点"三个字的真实含义。
- **B3 二十框之谜**：给 /detect 发一张带亮斑的图，看一个亮斑出十几个框。用你关3 的 `iou()` 手算相邻框 IoU（≈0.143 < 0.5 所以 NMS 压不掉）→ 把 iou_thr 调 0.1 重发。体会**框尺寸和 NMS 阈值是耦合参数**。

## 卡住问法模板

> "关N，我在写 XXX 函数，卡在 YYY（贴报错/贴你的代码）。只要提示不要答案。"

——这样我知道给多重的提示：先给方向，再追问再逼近。
