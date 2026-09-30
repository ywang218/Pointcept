# MiniLOps 真货训练链路 · 操作手册（Git Bash 版）

> 目标读者：未来的你。所有命令都在 **Git Bash** 里敲，工作目录一律是仓库根目录：
> ```bash
> cd /c/Users/GW00408524/Desktop/dev/Pointcept
> ```
> 所有 python 命令都要带 `-X utf8`（Windows 控制台默认 GBK，不带会中文乱码）。

> 🖥️ **新电脑 / 重新 clone 之后**：先看**第 7 节《新电脑复原清单》**——
> 哪些自动就有、哪些要补、哪些不用管，一张表说清。

---

## 0. 全景：这条链路里都有什么

```
数据（已有）              训练（一次性）                使用（日常）
─────────────           ──────────────────           ──────────────────
data_cache/             demo_real_train.py           demo_real_detect.py
  mini_train.json  ──►    基线评估 → 微调 → 复评  ──►  加载权重 → 对照片画红绿灯框
  mini_val.json            │
  images_train/            ├─► tl_finetuned.pth（微调权重，检测时自动用）
  images_val/              ├─► train_log.json（成绩单 + loss 曲线）
                          └─► eval_before/ eval_after/（前后对比图）
```

三个文件各管一段，和玩具版一一对应：

| 真货 | 干什么 | 玩具版对应 |
|---|---|---|
| `demo_real_train.py` | 训练闭环（forward+backward+参数更新） | `retrain_feedback.py` |
| `demo_real_detect.py` | 推理（对任意照片画框） | `demo_traffic_light.py` 的推理半段 |
| `data_cache/` | 数据（COCO 格式 360 张图） | `make_synthetic_frame` |

---

## 1. 训练：怎么跑、怎么盯、怎么算跑完

### 1.1 启动训练

```bash
# 完整闭环（CPU 约 25~30 分钟）
python -X utf8 minilops/demo_real_train.py

# 冒烟模式：3 图 1 轮（约 1 分钟）—— 改了代码先跑这个，确认链路不报错
python -X utf8 minilops/demo_real_train.py --quick
```

跑起来后终端会持续打印：读数据 → 加载模型 → 基线评估 → 每 10 步一行 loss → 复评 → 成绩单对比表。**不想盯屏幕就放着不管**，进度同时写进了文件（见 1.2）。

想在后台跑、终端继续干别的：

```bash
python -X utf8 minilops/demo_real_train.py > train_out.txt 2>&1 &
```

### 1.2 看进度（另开一个 Git Bash 窗口）

```bash
tail -f minilops/data_cache/train_progress.txt    # -f = 持续刷新，Ctrl+C 退出查看
# 或只看最后几行：
tail -3 minilops/data_cache/train_progress.txt
```

输出长这样，看 `step` 和 `ETA` 两个字段：

```
step   70/240   loss 0.247 (cls 0.065 box 0.163)   4.1s/步  ETA 11分
      ^^^^ ^^^                                    ^^^^^^^^^^
      当前步/总步数                                 还要多久
```

### 1.3 判断跑完了没

**跑完的标志是 `train_log.json` 出现**（它是最后一步才写的；没有 = 还在跑）：

```bash
ls minilops/data_cache/train_log.json 2>/dev/null && echo "✅ 跑完了" || echo "⏳ 还在跑"
```

同理这几个也是收尾产物：`tl_finetuned.pth`（权重）、`eval_after/`（对比图）。
⚠️ 训练跑着的时候**别再启动第二个训练**——两个进程写同一批文件会互相踩踏。

### 1.4 中途反悔

训练进程在前台就 `Ctrl+C`；后台跑的用 `kill %1`（或 `jobs` 看任务号再 kill）。中断不伤数据，重跑会从头来（覆盖写）。

---

## 2. 训练完看结果（三层，由浅入深）

### 2.1 第一层：数字成绩单

```bash
cat minilops/data_cache/train_progress.txt | tail -1   # 最后的 loss
python -X utf8 -c "import json; d=json.load(open('minilops/data_cache/train_log.json')); print('基线:', d['baseline']); print('微调后:', d['after'])"
```

看懂三个数（越大越好 / 越小越好）：

| 指标 | 含义 | 翻译 |
|---|---|---|
| precision | 报的框里多少是对的 | 误报多不多 |
| recall | 该找的框找回了多少 | 漏检多不多 |
| F1 | 两者的调和平均 | 总分，一眼定优劣 |
| FN | 漏掉的真值框数 | 微调最该吃掉的数字 |

### 2.2 第二层：肉眼对比图（最直观，强烈推荐）

```bash
explorer.exe minilops/data_cache/eval_before &   # 训练前的检测
explorer.exe minilops/data_cache/eval_after &    # 训练后的检测
```

两个文件夹里是**同 8 张图、同名文件**。每张图上：
- **绿框 = 真值**（人标的，标准答案）
- **红框 = 模型检测**（带置信度分数）

同名文件左右开窗对看：红框和绿框贴得越紧、数量越接近，训得越好。漏检（有绿没红）和误报（有红没绿）一眼可辨。

### 2.3 第三层：loss 曲线（训练过程本身）

```bash
python -X utf8 -c "
import json
d = json.load(open('minilops/data_cache/train_log.json'))
for s in d['steps'][::20]:
    print(f\"step {s['step']:3d}  total {s['total']:.3f}  cls {s['loss_classifier']:.3f}  box {s['loss_box_reg']:.3f}\")
"
```

`total` 一路震荡下行 = 真在学习（和玩具版 `demo_traffic_light.py` 训练段同款判据）。四项 loss 的含义见 `demo_real_train.py` 文件头注释。

---

## 3. 推理：用训好的权重检测任意照片

### 3.1 基本用法

```bash
# ① 处理 real_photos/ 里全部照片
python -X utf8 minilops/demo_real_detect.py

# ② 指定图片（任何 jpg/png 路径都行）
python -X utf8 minilops/demo_real_detect.py 你的照片.jpg

# ③ 显式指定用哪个权重
python -X utf8 minilops/demo_real_detect.py --ckpt minilops/data_cache/tl_finetuned.pth 照片.jpg
```

**权重自动选择**：`data_cache/tl_finetuned.pth` 存在就自动加载微调版（启动时会打印"已叠加微调权重"）；不存在就用官方预训练基线版。想强制对比两版：

```bash
# 基线版（临时躲开微调权重：--ckpt 指向一个不存在的路径）
python -X utf8 minilops/demo_real_detect.py --ckpt /dev/null 照片.jpg
```

输出：控制台一张检测表（全部类别按分数排序）+ 红绿灯专项行，并在照片旁边生成 `xxx_detected.jpg`（红粗框=红绿灯，灰细框=其他物体）。

### 3.2 拿 val 集里的图自己试（有真值可对照）

```bash
# 随便挑几张 val 图做推理输入
python -X utf8 minilops/demo_real_detect.py minilops/data_cache/images_val/000000171382.jpg
```

和 `eval_before/000000171382_gt.jpg`（真值可视化）对看，就是一次迷你评测。

### 3.3 常见问题

| 症状 | 原因 / 解法 |
|---|---|
| 中文/emoji 乱码 | 忘了 `-X utf8` |
| `No such file or directory: ...tl_finetuned.pth` | 还没训练过 → 先跑第 1 节；或忽略（会退回基线版） |
| 首次运行卡在下载 | 在下官方权重 ~160MB，一次性，之后走本地缓存 |
| 检测不到红绿灯 | 分数阈值 SCORE_THR=0.5 太严 → 看控制台"最高分仅 x.xx"提示，或调低阈值重跑 |
| 想换张照片试试 | 直接把图丢进 `minilops/real_photos/` 再跑 ① |

---

## 4. 文件清单（训练跑完后 data_cache/ 全家福）

| 文件/目录 | 谁生成的 | 干什么用 |
|---|---|---|
| `mini_train.json` / `mini_val.json` | 建数据集时 | COCO 格式标注（300 训 / 60 测） |
| `images_train/` / `images_val/` | 建数据集时 | 图片本体 |
| `train_progress.txt` | 训练中实时写 | 盯进度 |
| `tl_finetuned.pth` | 训练结束 | **微调权重**，检测时自动加载 |
| `train_log.json` | 训练结束 | 成绩单（baseline/after）+ 240 步 loss 曲线 |
| `eval_before/` `eval_after/` | 训练中评估时 | 前后对比图（绿=真值 红=检测） |
| `annotations*/` | 建数据集时 | COCO 官方原始标注（242MB，已用不到，可删） |

---

## 5. 学习路线挂钩（回头深究时用）

这份手册管"怎么跑"；"为什么这么写"在代码注释里，按这个顺序抠：

1. `demo_real_train.py` 文件头 —— 玩具版↔真货版五步心跳对照表
2. `load_split()` —— 上一关作业的参考答案（xywh→xyxy 坑在这）
3. `finetune()` —— ①zero→②forward→③loss→④backward→⑤step 每行都有注释
4. `iou_matrix()` / `match_boxes()` —— my_d3 要写的 NMS 的表亲（贪心配对）
5. `evaluate_tl()` —— `retrain_feedback.evaluate()` 的真货版（P/R/F1）

改训练参数做实验：`demo_real_train.py` 顶部"训练旋钮"区（LR / EPOCHS / MAX_TRAIN / FREEZE_BACKBONE），改完先 `--quick` 冒烟再全量。

---

## 6. 换考卷：S2TLD 交大小红绿灯（域差距实验）

> 背景：COCO val 是预训练权重"刷过的原题"，F1 0.56~0.58 只是复习分。真实本事要在
> **从没见过的考卷**上验——S2TLD（上海交大 2022 发布，行车记录仪视角，远处小红绿灯
> 平均 21×49 像素），托管在**魔搭 ModelScope（阿里国内站，直连无墙）**。

### 6.1 下载数据（一次性，约 1.4GB，下载几分钟）

```bash
cd /c/Users/GW00408524/Desktop/dev/Pointcept
mkdir -p minilops/data_cache/s2tld
cd minilops/data_cache/s2tld

# ① 标注包（0.15MB，秒下）：DSDL 格式答案
curl -L -o dsdl.zip \
  "https://modelscope.cn/datasets/OmniData/S2TLD/resolve/master/dsdl/dsdl_Det_full.zip"
unzip -o dsdl.zip && rm dsdl.zip

# ② 图片大包（1.4GB）：后台下，进度条照常显示
curl -L -o S2TLD.tar.gz \
  "https://modelscope.cn/datasets/OmniData/S2TLD/resolve/master/raw/S2TLD.tar.gz" &
```

**大包下完的标志**（`ls -la S2TLD.tar.gz` 显示约 **1.42 GB / 1424877462 字节** 即完整）：

```bash
ls -la minilops/data_cache/s2tld/S2TLD.tar.gz
# -rw-r--r-- ... 1424877462 ... S2TLD.tar.gz   ← 这个字节数 = 下完了
```

### 6.2 解压（约 1~2 分钟）

```bash
cd minilops/data_cache/s2tld
tar -xzf S2TLD.tar.gz        # 解出 S2TLD/S2TLD（720x1280）/normal_*/JPEGImages/...
```

解压完跑一下检查（该有的都在）：

```bash
ls minilops/data_cache/s2tld/dsdl_Det_full/set-720x1280/   # 应有 720x1280_samples.json
ls "minilops/data_cache/s2tld/S2TLD/S2TLD（720x1280）"      # 应有 normal_1  normal_2
```

**目录名注意**：真实目录带**全角括号** `S2TLD（720x1280）`（打命令时直接复制上面这行
最稳）。脚本里用 glob 模糊匹配 `*720x1280*` 自动兜住全角/半角变体，不用你手工改名。

### 6.3 跑考试（先冒烟再正式）

```bash
cd /c/Users/GW00408524/Desktop/dev/Pointcept

# 冒烟：20 张（1~3 分钟）—— 先确认链路无报错
python -X utf8 minilops/demo_s2tld_eval.py --quick

# 正式：150 张（CPU 约 5~10 分钟），权重自动用 tl_finetuned.pth（微调版）
python -X utf8 minilops/demo_s2tld_eval.py

# 也可以指定考多少张 / 用哪个权重：
python -X utf8 minilops/demo_s2tld_eval.py 60
python -X utf8 minilops/demo_s2tld_eval.py --ckpt /dev/null   # 强制 COCO 基线版（对比用）
```

跑到一半想看进度：另开窗口 `tail -f` 没有进度文件——但控制台每 10 张打一行
`图 xx/150 累计 TP … ETA x分`，盯着跑就行。

### 6.4 看结果（三层，和第 2 节同款）

```bash
# ① 数字成绩单（跑完随时重看）
python -X utf8 -c "
import json
d = json.load(open('minilops/data_cache/s2tld_result.json', encoding='utf-8'))
print('S2TLD:', d['s2tld'])
print('对照 COCO:', d['coco_compare'])
"

# ② 肉眼对比图（绿=真值，红=检测）—— 看"漏检"有多惨就开这个
explorer.exe minilops/data_cache/s2tld_eval &

# ③ 跑的过程中控制台最后会打"同一个模型，两张考卷"并排对比
```

### 6.5 预期结果与含义

| 考卷 | 模型见没见过 | 预期 F1 |
|---|---|---|
| COCO val（基线） | 见过原题（同源） | ≈ 0.58 |
| COCO val（微调后） | 见过原题（同源） | ≈ 0.56 |
| **S2TLD** | **从没见过（2022 新域）** | **大幅跳水，可能接近 0** |

- 惨是**正常的**——这一课的课题就叫 **domain gap（域差距）**：灯太小（平均 21×49
  像素 vs COCO 里的几十~几百像素）、视角不同（行车第一视角 vs 街景）、光照条件刁钻。
- 成绩单里还有两个"惨状诊断"字段：`img_with_det`（多少张图**至少**检出 1 个灯）
  和 `best_score`（全场最高红绿灯分）——如果 best_score 连 0.5 都不到，说明不是
  阈值卡死的，是模型真的"看不见"这种小灯。
- **这正是产线"数据闭环"的终极答案**：模型在训练分布里再好，出了分布就现形。
  下一关（S2TLD 微调救分）就是闭环的下半场。

---

## 7. 新电脑复原清单（clone 之后照此复活整条链路）

> 背景：git 仓库只收代码 + 小数据。大块头（1.4GB 数据集 / 160MB 权重）被
> `.gitignore` 挡在外面（GitHub 单文件 100MB / 单次 push 2GB 硬限制）。
> 本节回答"clone 下来缺什么、怎么补"。**把本节原文贴给那台电脑上的 AI 助手即可。**

### 7.0 环境准备（一次性）

```bash
# 需要的包（python 3.11 + torch/torchvision CPU 版 + PIL + numpy）：
pip install torch torchvision pillow numpy
# 玩具版链路（run_all.py / server.py）另需：
pip install flask
```

⚠️ 本仓库 Pointcept 本体的一堆依赖**不用装**——minilops 的 demo 全部零依赖独立可跑，
torch 是唯一的大件。首次跑 detect 会自动下载 torchvision 官方预训练权重（~160MB，
一次性，走本地缓存），需联网。

### 7.1 缺什么、补什么（总表）

| clone 后的状态 | 是否要手动补 | 怎么补 |
|---|---|---|
| 代码、RUNBOOK、README | ✅ 自动就有 | — |
| `mini_train.json` `mini_val.json` `images_val/` | ✅ 自动就有 | — |
| `images_train.zip`（49MB） | ✅ 自动就有 | 解压：`cd minilops/data_cache && unzip -o images_train.zip` |
| `tl_finetuned.pth`（160MB 微调权重） | ❌ 要补 | `python -X utf8 minilops/demo_real_train.py` 重训（CPU 25~30 分钟） |
| `s2tld/`（1.4GB 数据集） | ❌ 要补（只有做第 6 节才需要） | 见 7.3 |
| `annotations/`（COCO 原始标注 242MB） | 不用补 | RUNBOOK §4 已标"已用不到，可删" |
| `eval_before/` `eval_after/` `train_log.json` 等 | 不用补 | 重训时自动再生 |

> ⚠️ 若 clone 后 `data_cache/` 整个不存在或解压后 json 找不到图，先确认
> checkout 的分支含 "self training round trip" 这个 commit（`git log --oneline -5` 查）。

### 7.2 复原训练成果（权重 + 成绩单）

```bash
cd /c/Users/你/Desktop/dev/Pointcept     # 换成你的仓库根目录
cd minilops/data_cache && unzip -o images_train.zip && cd ../../..   # ① 解压训练图
python -X utf8 minilops/demo_real_train.py --quick   # ② 冒烟 1 分钟，链路通就继续
python -X utf8 minilops/demo_real_train.py           # ③ 正式重训（CPU 25~30 分钟）
```

跑完 `tl_finetuned.pth` / `train_log.json` / `eval_before|after/` 全部回来，
之后推理、S2TLD 考试照第 3、6 节原样跑。**没有它也能跑推理**——detect 会
自动退回官方基线权重（就是分低一截，正好复现 §6.5 的对比实验）。

### 7.3 复原 S2TLD 数据集（只在要跑第 6 节时）

照抄 **§6.1 下载 + §6.2 解压**（魔搭 ModelScope，国内直连）：
`dsdl.zip`（0.15MB）+ `S2TLD.tar.gz`（1.4GB，**完整字节数 1424877462**，§6.1 有核对命令）。

### 7.4 验收（跑通 = 复原成功）

```bash
python -X utf8 minilops/demo_real_detect.py minilops/data_cache/images_val/000000002006.jpg
# 出检测表 + 生成 *_detected.jpg → 数据 OK
python -X utf8 -c "import json; print(json.load(open('minilops/data_cache/train_log.json', encoding='utf-8'))['after'])"
# 打出指标 → 权重 OK（重训后 F1 应回到 0.55 上下）
```

> 为什么权重不进 git：160MB 超 GitHub 单文件 100MB 限制（物理推不上去）；
> 且它可由数据+代码 30 分钟确定性再生，git 里放代码+数据、产物落磁盘，正是产线
> "代码进版本库、工件进工件库"的分工。

