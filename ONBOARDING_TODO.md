# Pointcept 上手 TODO
# + 今晚执行版（针对：懂 DL 基本概念、懂 Transformer 原理、无实践、前端图形渲染背景）

这个仓库主要是 3D 点云感知框架，核心能力在语义分割、实例分割、预训练和多模态数据管线。
如果你的近期待办是 `LiDAR seg`，这份清单可以直接照着走。
如果你的近期待办是 `2D 检测/分割`，要先确认它是不是需要外部 2D 模型栈，Pointcept 更像训练和数据流框架，不是完整的 2D detection repo。

## 0. 领导那段话，实际在说什么
- [ ] `封装成算子集成到工作流`：把模型做成稳定的推理入口，重点是输入输出协议、批处理、模型加载、失败处理、耗时和可复现性。
- [ ] `参与这些算子模型的训练`：把训练数据、config、loss、评估和 checkpoint 跑顺，必要时继续做微调/重训。
- [ ] `LiDAR seg 预研`：先把点云分割的数据链路和 baseline 跑明白，验证是否值得继续投入高质量标注。
- [ ] `红绿灯 2D 预标注`：这更像生产侧任务，通常是 2D 检测/分割模型 + 产线封装，不一定由 Pointcept 原生覆盖。

## 0.1 这些事在仓库里对应什么
- [ ] 推理/算子封装：`tools/test.py`、`pointcept/engines/test.py`、`pointcept/models/default.py`、`pointcept/utils/checkpoint.py`
- [ ] 训练链路：`tools/train.py`、`pointcept/engines/train.py`、`pointcept/engines/defaults.py`
- [ ] 数据处理：`pointcept/datasets/*`、`pointcept/datasets/preprocessing/*`
- [ ] 模型和配置：`pointcept/models/*`、`configs/*`
- [ ] 自定义算子/CUDA 扩展：`libs/*`

## 1. 先把框架骨架看懂
- [ ] 看 `tools/train.py`，搞清楚训练入口怎么走到 trainer。
- [ ] 看 `tools/test.py`，搞清楚推理入口和训练是否共用配置。
- [ ] 看 `pointcept/engines/defaults.py`，理解 config 解析、分布式启动、seed、save_path。
- [ ] 看 `pointcept/engines/train.py`，理解 `Trainer -> dataset -> model -> optimizer -> scheduler -> hooks` 的主循环。
- [ ] 看 `pointcept/utils/registry.py`，理解 `type=...` 是怎么实例化出来的。

## 2. 先跑通一个 baseline
- [ ] 选一个最接近你业务的数据集先跑通。
- [ ] 如果你做 LiDAR seg，优先看 `configs/nuscenes/semseg-pt-v3m1-0-base.py` 或 `configs/semantic_kitti/semseg-pt-v2m2-0-base.py`。
- [ ] 先只追求"能启动、能训练、能保存 checkpoint"，不要一开始就改模型。
- [ ] 跑一次 `train.sh`，确认 `exp/` 目录、日志、tensorboard、权重都正常生成。
- [ ] 跑一次 `test.sh`，确认评估流程和输出格式。

## 3. 把数据链路吃透
- [ ] 看 `README.md` 里的 `Installation`、`Data Preparation`、`Quick Start`。
- [ ] 看 `pointcept/datasets/defaults.py`，理解原始样本是怎么被读成 `coord/color/strength/segment` 的。
- [ ] 看你目标数据集对应的 dataset 类。
  - [ ] `pointcept/datasets/nuscenes.py`
  - [ ] `pointcept/datasets/semantic_kitti.py`
  - [ ] `pointcept/datasets/waymo.py`
- [ ] 看 `pointcept/datasets/transform.py`，理解随机增强、采样、收集字段是怎么串起来的。

## 4. 把模型链路吃透
- [ ] 看 `pointcept/models/__init__.py`，先知道仓库里有哪些模型家族。
- [ ] 看 `pointcept/models/default.py`，理解默认 segmentor / classifier 的输入输出约定。
- [ ] 打开一个具体 config，例如 `configs/nuscenes/semseg-pt-v3m1-0-base.py`。
- [ ] 对照 config 里的 `model / data / optimizer / scheduler / hooks`，搞清楚每一块改动会影响什么。

## 5. 对齐你的业务任务
- [ ] 如果是 `LiDAR seg`，先确定数据源是 `nuScenes`、`SemanticKITTI` 还是 `Waymo`。
- [ ] 如果是 `2D 预标注`，先确认 Pointcept 在你们流程里只负责哪一段。
- [ ] 如果是"封装成算子接入工作流"，先定义清楚输入、输出、batch、设备、失败重试、模型加载方式。
- [ ] 如果是"参与训练"，先确认新增数据集、损失、后处理能不能通过 registry 接进去。

## 6. 最后再做工程化收口
- [ ] 能用一句话说清楚训练/测试的数据流。
- [ ] 能独立改一个 config 并解释改动效果。
- [ ] 能新增一个 transform 或 dataset。
- [ ] 能把推理封装成一个稳定的服务/算子接口。
- [ ] 能把一次实验结果固定成可复现的命令和配置。

---

# ══════════════════════════════════════════════════════
# 今晚执行计划（约 3 小时，全部本机可验证，无需 GPU）
# 前提：你懂 DL 基本概念 + Transformer 原理，缺的是"点云领域 + PyTorch 工程"
# ══════════════════════════════════════════════════════

## 今晚的定位

你不需要再学"什么是 attention"。你缺的是三样：
1. **点云领域的表示约定**（无序、无网格、offset 分 batch——和 NLP/CV 的序列/网格完全不同）
2. **一个真实 Transformer 变体长什么样**（PTv2 的 GroupedVectorAttention 和教科书 attention 的三点差异）
3. **工程骨架**（Registry/config/hook——这个你看 10 分钟就懂，和前端组件系统同构）

时间分配：**动手验证 90 分钟 > 读代码 60 分钟 > 读文档 30 分钟**。你是"懂原理缺实践"的类型，每个概念都必须亲手跑一遍才真正落地。

## Step 0：环境准备（15 分钟）

```bash
pip install numpy open3d addict pyyaml
```
（不需要 torch——今晚的验证实验全部不依赖 GPU 框架）

## Step 1：亲手"摸"一次点云（20 分钟）——建立领域体感

运行：
```bash
python -c "
import numpy as np, open3d as o3d
# 模拟一帧 LiDAR：一帧 SemanticKITTI 约 12 万个点
n = 120000
coord = np.random.randn(n, 3) * [30, 30, 2]   # 车辆周围 60m x 60m x 4m
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)
pcd.paint_uniform_color([0.5, 0.7, 1.0])
o3d.visualization.draw_geometries([pcd])       # 弹出可旋转的 3D 窗口
"
```
看的时候想一个问题：**这团点没有 index buffer、没有拓扑、没有固定的点数**——那你熟悉的"网格上的卷积"在这里要怎么做？这就是整个领域要解决的第一问题。

## Step 2：亲手做一次 GridSample（30 分钟）——今晚最重要的实验

这就是 PTv2 整个模型的骨架概念（`grid_sizes=(0.06, 0.12, 0.24, 0.48)` 就是四级这样的操作）。
运行并观察输出（**注意用 `python -X utf8` 跑，避免 Windows 控制台中文乱码**）：
```bash
python -X utf8 -c "
import numpy as np
n = 120000
rng = np.random.default_rng(42)
# 模拟一帧 64 线 LiDAR：极坐标扫描，近密远疏
angles = rng.uniform(0, 2*np.pi, n)
r = np.sqrt(rng.uniform(0, 1, n)) * 35
x = r*np.cos(angles); y = r*np.sin(angles); z = rng.uniform(-2, 3, n)
coord = np.stack([x, y, z], axis=1)
for gs in [0.05, 0.06, 0.12, 0.24, 0.48]:   # PTv2 四级下采样的真实 grid_size
    grid = np.floor(coord / gs).astype(np.int64)
    key = grid[:,0]*10**10 + grid[:,1]*10**5 + grid[:,2]
    print(f'grid_size={gs}: {n} -> {len(np.unique(key))}')
"
```
我已替你验证过，输出大约是（随机合成数据，真实 LiDAR 去重率会更高）：
```
grid_size=0.05: 120000 -> 119959
grid_size=0.06: 120000 -> 119928
grid_size=0.12: 120000 -> 119354
grid_size=0.24: 120000 -> 115116
grid_size=0.48: 120000 ->  87774
```
注意：**合成随机点去重不明显（点本来就稀疏）；真实 LiDAR 一帧约 12 万点在 0.05m 格子下会显著去重到几万**。做实验时把 `r` 的范围改小（如 `* 10`）就能看到剧烈的合并效果——点密度上去了，同一格子里挤多个点的情况就多了。
对照仓库代码：`pointcept/datasets/transform.py` 的 `GridSample`（840 行起）——它的 `fnv_hash_vec` 就是上面那个 key 的工业版（防溢出哈希）。config 里 `grid_size=0.05`、训练模式每个 voxel 随机取一个点、`return_grid_coord=True` 把整数格坐标存下来给模型当"token 位置"用。

**理解检查（写在笔记本上）**：
- [ ] Q1: 为什么点云网络一定要先做 GridSample？（答：控制显存 + 把无序点变成"格点 token"，给 attention 提供邻域结构）
- [ ] Q2: grid_size 变大 2 倍，点数大约怎么变？（跑一下就知道）

## Step 3：读 PTv2 的 attention，找出它和标准 Transformer 的 3 个差异（40 分钟）

> **前置阅读（重要）**：先读 `step2_补充讲解C_QKV纠正与预处理动机.md` ——
> 它纠正了 QKV 的常见误解（Q/K/V 是同一 token 的三份并列投影；权重是 softmax(Q·Kᵀ) 算的；V 是被取材的"货物"），
> 并确认了"预处理（GridSample→token 化、kNN→邻域化）就是为了 attention 可行"的动机。
> 读完这份再读代码，三个差异点会一眼即懂。

你已经懂 QKV/softmax/加权平均。打开：
`pointcept/models/point_transformer_v2/point_transformer_v2m2_base.py`

**先看 forward 主流程（556 行，20 行读完）**：
```python
points = self.patch_embed(points)          # 原始点 -> 48 维 token
for i in range(4):
    points, cluster = self.enc_stages[i](points)   # GridPool 下采样 + attention
    skips.append(...)                       # 记录 skip（U-Net 结构！）
for i in reversed(range(4)):
    points = self.dec_stages[i](...)        # 上采样 + skip 融合
seg_logits = self.seg_head(feat)            # 每点 -> 19 类 logits
```
这就是**体素化 U-Net + 每级 Transformer block**。你熟悉的 LOD/clipmap 结构直接复用这个直觉。

**再看 GroupedVectorAttention.forward（103 行）**，带着三个问题找答案：
- [ ] D1: 标准 Transformer 对全序列做 attention；这里 `reference_index`（kNN 邻居表）限制了每个点只看多少个邻居？（这叫局部注意力——点云太大，全局算不起）
- [ ] D2: 标准 Transformer 用加性位置编码（正弦/可学习）；这里 `linear_p_bias(pos)` 用什么当位置信息？（答：邻居的**相对 3D 坐标差**，`pos = key[:,: ,0:3]` 是 grouping 带回来的 xyz）
- [ ] D3: 标准注意力 softmax 在整个 key 维度上做一次；这里 `groups` 把 channel 分成 G 组，每组独立 softmax（`einops.rearrange('n ns (g i) -> n ns g i')`），为什么？（答：向量注意力——每个通道组有自己的权重分布，比单标量权重表达力强）

**配一个 5 行 CPU 验证（可选，装 torch 的话）**：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "
import torch, einops
q = torch.randn(4, 32)            # 4 个点、32 维特征
idx = torch.tensor([[0,1,-1,-1],[1,2,3,-1],[2,0,-1,-1],[3,2,1,-1]])  # kNN 邻居，-1=padding
k = torch.randn(4, 4, 32); v = torch.randn(4, 4, 32)   # 已按邻居gather的 K V
w = torch.softmax((k - q.unsqueeze(1)).norm(dim=-1, keepdim=True) * -1, dim=1)
out = (w * v).sum(1)               # 每个点 = 邻居加权平均
print('局部 attention 输出:', out.shape)   # [4, 32]，和输入同形
"
```
这就是"局部注意力"的最小骨架——PTv2 的第 103~129 行是它的工业版（加分组、位置编码、CUDA 加速）。

## Step 4：工程骨架速通（30 分钟）——你的主场，快速收割

- [ ] `pointcept/utils/registry.py`（15 分钟）：`@MODELS.register_module()` 装饰器 + `build(dict(type="PT-v2m2", ...))`。心智模型：Vue 的 `app.component()` + `<component :is>`。
- [ ] 验证（本机可跑）：
  ```bash
  cd /c/Users/GW00408524/Desktop/dev/Pointcept
  pip install addict pyyaml  # 已装则跳过
  python -c "from pointcept.models.builder import MODELS; print(len(MODELS.module_dict), '个模型已注册')"
  ```
  （如果 import 报 torch 缺失——说明 models/__init__ 链上有 torch 依赖，那就改读 `pointcept/utils/registry.py` 源码本身，纯 python 无依赖，10 分钟能读完）
- [ ] `pointcept/engines/train.py` 主循环（15 分钟）：只看 `train_step`（195~230 行）——前向 → loss.backward() → optimizer.step()，外面是 hook 生命周期（before_train / after_epoch...）。心智模型：**生命周期钩子模式**，CheckpointSaver/InformationWriter 全是插件。
- [ ] 打开 `configs/_base_/default_runtime.py` 看 hooks 列表，对照 `pointcept/engines/hooks/misc.py` 的类名——config 字符串 → hook 实例，Registry 又一次出现。

## Step 5：读 config，做"翻译练习"（20 分钟）

打开 `configs/semantic_kitti/semseg-pt-v2m2-0-base.py`，**逐块翻译成中文**（写在笔记里）：
- [ ] `batch_size=8, mix_prob=0.8, enable_amp=True` 是什么
- [ ] `model` 块：`grid_sizes=(0.15,...)` 对应你 Step 2 做的事；`num_classes=19` 对应文件里的 19 个类名
- [ ] `criteria`：CrossEntropyLoss（带每类权重，19 个数）+ LovaszLoss（直接优化 mIoU 的损失，点云分割标配）
- [ ] `data.train.transform` 列表：**按顺序**读一遍——RandomRotate/Scale/Flip/Jitter（数据增强）→ GridSample（Step 2）→ PointClip（裁到 70m×70m 感知范围）→ SphereCrop（限制最多 12 万点）→ ToTensor → Collect（决定哪些字段进模型）
- [ ] `scheduler=OneCycleLR`：lr 从 0.0002 热身到 0.002 再余弦退火到接近 0

这个练习做完，"改一个 config 并解释改动效果"（TODO 第 6 节）就达成一半了。

## Step 6：收尾（15 分钟）

- [ ] 把 Step 2/3 的理解检查答案写进本文件（或笔记）
- [ ] 用一句话写下今晚的数据流（示例答案在下方，先自己写再看）：
  > `.bin 文件读出 [N,4] → 增强扰动 → GridSample 体素化(0.05m)取代表点 → [N',C] token + 格坐标 → PTv2: patch_embed + 4级(attention+GridPool下采样) + 4级上采样(skip) → 每点 19 类 logits → CE+Lovasz loss → backward → AdamW 更新 → hook 存 checkpoint/记 wandb`
- [ ] 明天要问同事/领导的两件事：
  1. **GPU 资源在哪**（本机 RTX 500 Ada 4GB 只够推理验证，跑不动训练；train.sh 里有 SLURM 分支，问是不是有集群）
  2. **LiDAR seg 的数据源**是 nuScenes/SemanticKITTI/Waymo 还是自有数据（决定你先准备哪个 dataset 类）

## 今晚不做什么（同样重要）

- ❌ 不装 CUDA、不编译 `libs/pointops`（Windows 上坑极深，且推理/训练在服务器上做）
- ❌ 不下载 SemanticKITTI（80GB，本机磁盘只有 71GB 剩余，放不下）
- ❌ 不看 PTv3/Sonata/Concerto 等其他模型家族（等 baseline 跑通再说）
- ❌ 不深究分布式 launch（DDP 是服务器上的事）

## 环境事实（已核实）

| 项 | 状态 |
|----|------|
| GPU | RTX 500 Ada **4GB**（推理可，训练不够） |
| conda/WSL/Docker/nvcc | **都没有** |
| 磁盘剩余 | 71GB（SemanticKITTI 约 80GB，放不下） |
| Python | 3.11.9 + pip 可用 |
| 仓库状态 | git clone 完整，`exp/` 不存在（还没跑过任何实验） |

## 本周后续（预告，今晚不用做）

1. 装一个 CPU torch，用 `DefaultSegmentor` + 假数据跑一次前向（验证模型链路理解）
2. 拿一帧公开样例点云（SemanticKITTI 官网有样例），用 open3d 上色渲染，配合 `visualization.py` 的 `get_point_cloud`
3. 问清楚 GPU 后：服务器上按 README Method 1 装环境 → 跑通 `semseg-pt-v2m2-0-base`（TODO 第 2 节）
