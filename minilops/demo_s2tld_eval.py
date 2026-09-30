# -*- coding: utf-8 -*-
"""
📉 换考卷 —— S2TLD 交大小红绿灯：考出模型的"域差距"
=====================================================
【为什么要换考卷】前面所有成绩单都有隐藏水分：COCO val 和预训练数据同源同分布
——考它等于让学霸重做做过的卷子，F1 0.56 是"复习分"。真本事要在
【从没见过的考卷】上验。本文件请出 S2TLD（上海交大 & 安徽大学，2022 发布）：

    旧考卷（COCO val）                  新考卷（S2TLD 720x1280 子集）
    ───────────────────────────       ─────────────────────────────────
    各国街景照片（多为行人视角）        一辆车的行车记录仪第一视角（上海街景）
    红绿灯近、大、画面里显眼           远处小灯：平均 21×49 像素（宽×高）
    预训练权重当年刷过的原题            2022 年才发布，预训练从没见过
    框/图 3.5 个，F1 ≈ 0.56~0.58       ？← 本文件揭晓

数据托管在魔搭 ModelScope（阿里国内站，直连无墙），下载见 RUNBOOK 第 6 节。

【新格式课】这是你遇到的第三种标注格式——
    COCO json   三张表 + 外键拼装（images/annotations/categories）
    Bosch yaml  一图一条的清单（本次已弃，官方服务器挂了）
    S2TLD json  samples 列表，media_path + annotations 内嵌，bbox 是 xywh
  ↑ 注意：bbox 又是 xywh！COCO 作业里那个坑（xywh→xyxy 要 += 换算）原样再现。
  真实产线就是天天面对不同格式的到货数据——"先验格式再动手"是铁律。

【实验公平性】尺子完全不变：SCORE_THR=0.5、IoU 匹配=0.5、同一套配对算法——
变得只有考卷。一次实验只动一个变量，两张考卷的分数才可比。

【和前面文件的关系】
    数据段   load_s2tld()      —— json 版 load_split（第三种格式的参考答案）
    评估段   iou/match/评分    —— 和 demo_real_train.py 同款（复制保持独立可跑）
    模型段   权重自动选择      —— 和 demo_real_detect.py 同款（微调版优先）

跑法（仓库根目录）：
    python -X utf8 minilops/demo_s2tld_eval.py             # 考 150 张（CPU 约 5~10 分钟）
    python -X utf8 minilops/demo_s2tld_eval.py 60          # 自定张数
    python -X utf8 minilops/demo_s2tld_eval.py --quick     # 冒烟 20 张（1~3 分钟）
    python -X utf8 minilops/demo_s2tld_eval.py --ckpt xx   # 指定权重（--ckpt /dev/null = 强制基线版）

数据准备（详见 RUNBOOK 第 6 节）：
    minilops/data_cache/s2tld/dsdl_Det_full/set-720x1280/720x1280_samples.json   ← 答案
    minilops/data_cache/s2tld/S2TLD（720x1280）/normal_*/JPEGImages/*.jpg         ← 考卷

输出（都在 data_cache/ 下）：
    s2tld_eval/          前 8 张的可视化（绿=真值，红=检测）
    s2tld_result.json    成绩单（跑完随时重看，命令见 RUNBOOK 6.4）
    控制台               考试进度 + 最终成绩单（和 COCO 成绩并排对比）
"""

import io
import sys
import json
import time
from pathlib import Path
from collections import Counter

# Windows 控制台默认 GBK：包成 UTF-8 输出流（系列文件同款）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision

# ── COCO 类别表（和 demo_real_detect.py 同一张，复制进来保持文件独立可跑）──
COCO_CLASSES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
TL_ID = COCO_CLASSES.index("traffic light")   # = 10（用 index() 现算，不手数）

# S2TLD 的 5 个状态类别（category_id → 名字）。
# 考"检测"不考"认状态"：5 态全部算"红绿灯"这一类（和 Bosch 考卷同款策略）
S2TLD_STATES = {1: "红", 2: "黄", 3: "绿", 4: "关", 5: "等待"}

BASE = Path(__file__).resolve().parent / "data_cache"
S2TLD_ROOT = BASE / "s2tld"
SAMPLES_JSON = S2TLD_ROOT / "dsdl_Det_full" / "set-720x1280" / "720x1280_samples.json"

# ── 考试旋钮 ──
SCORE_THR = 0.5     # 分数阈值：和旧考卷同一把尺子，跨考卷才可比
IOU_MATCH = 0.5     # IoU 匹配阈值：同上
EVAL_N = 150        # 考多少张（720x1280 子集共 4564 张；150 张已够看清惨状）
N_SAVE = 8          # 存几张可视化
CKPT = BASE / "tl_finetuned.pth"   # 默认：有微调权重用微调版（demo_real_detect 同策略）

# 手写参数解析（系列同款风格）：--quick / --ckpt 路径 / 裸数字 = 考几张
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a == "--quick":
        EVAL_N, N_SAVE = 20, 4
    elif a == "--ckpt" and _i + 1 < len(_args):
        CKPT = Path(_args[_i + 1])
        _i += 1                     # --ckpt 连着吃掉下一个参数
    elif a.isdigit():
        EVAL_N = int(a)
    _i += 1
QUICK = "--quick" in _args


def load_font(size: int = 20):
    """Windows 自带粗体 Arial；找不到退回 PIL 默认小字体（系列文件同款）。"""
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = load_font()


# ─────────────────────── [0] 数据段（第三种格式的 load_split） ───────────────────────

def find_image_root() -> Path:
    """定位图像根目录。解压后的目录名带全角括号"S2TLD（720x1280）"，
    不同解压工具可能给出全角/半角/下划线变体——用 glob 模糊匹配兜住（*720x1280*），
    顺便兼容"tar 里多包了一层 S2TLD/"的情况。"""
    for cand in (S2TLD_ROOT, S2TLD_ROOT / "S2TLD"):
        hits = sorted(cand.glob("*720x1280*")) if cand.exists() else []
        if hits:
            return hits[0].parent
    return S2TLD_ROOT     # 都没找到 → 返回默认，让后面的 fail-fast 报完整提示


def load_s2tld(samples_json: Path, img_root: Path):
    """S2TLD samples.json → (records, 图像根目录实际用哪个)。records 每项：
       {"path": 图片路径, "boxes": (N,4) float32 xyxy, "states": (N,) int}

    和 demo_real_train.load_split(COCO json) 同一个岗位、第三种进货格式：
      COCO：三张表靠外键拼——要自己 GROUP BY
      S2TLD：samples 列表天然"一图一条"——不用拼表，但 bbox 是 xywh 要换算
    """
    data = json.load(open(samples_json, encoding="utf-8"))
    records = []
    for s in data["samples"]:               # 每条：{media: {media_path, ...}, annotations: [...]}
        raw = s.get("annotations") or []
        # 坏框过滤：宽/高 ≤ 0 的框画不出来也配不上对，扔——真实数据就是有脏数据
        keep = [a for a in raw if a["bbox"][2] > 0 and a["bbox"][3] > 0]
        if not keep:
            continue                        # 没灯的图考不出漏检，跳过
        # xywh → xyxy：前两列（左上角）不动，后两列 += 前两列——COCO 作业的坑原样再现！
        #   实打实（数据集里真实的一条）：[190.0, 126.0, 27.0, 57.0]
        #     → [190.0, 126.0, 190+27=217.0, 126+57=183.0]
        boxes = np.array([a["bbox"] for a in keep], dtype=np.float32)
        boxes[:, 2:] += boxes[:, :2]
        records.append({
            "path": img_root / s["media"]["media_path"],
            "boxes": boxes,
            # category_id 原样保留（1~5 → 红黄绿关等待），统计时再翻译
            "states": np.array([a["category_id"] for a in keep], dtype=np.int64),
        })
    return records


def coco_tl_size():
    """顺手量一下旧考卷：mini_val.json 里红绿灯的平均宽高（对比"新考卷的灯有多小"）。
    返回 (平均宽, 平均高, 灯数)；文件不在就返回 None（彩蛋不是硬依赖）。"""
    p = BASE / "mini_val.json"
    if not p.exists():
        return None
    data = json.load(open(p, encoding="utf-8"))
    # COCO 标注是 xywh：bbox[2]=宽、bbox[3]=高——不用换算，直接就是尺寸
    tl = [a["bbox"] for a in data["annotations"]
          if a["category_id"] == TL_ID and not a.get("iscrowd", 0)]
    if not tl:
        return None
    wh = np.array(tl, dtype=np.float32)[:, 2:]      # (N,2)：列切片，纯搬运族
    return float(wh[:, 0].mean()), float(wh[:, 1].mean()), len(tl)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL 图 → (3,H,W) float32 [0,1]（系列同款：复制→换轴→除 255）。"""
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


# ─────────────────────── [1] 评估工具段 ───────────────────────
# ↓ iou_matrix / match_boxes 复制自 demo_real_train.py（同款算法，保持文件独立可跑）

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组 xyxy 框的两两 IoU → (len(a), len(b))。IoU = 交集面积/并集面积（重叠率）。
    实打实：a=[[0,0,10,10]], b=[[5,5,15,15]] → 交 5×5=25，并 100+100-25=175 → 0.143
    a[:, None, 2] 把 (Na,4) 抬成 (Na,1,4)，和 b 的 (1,Nb,4) 广播成对 → (Na,Nb)。
    """
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)      # 负交长钳 0 = 不相交
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def match_boxes(preds: np.ndarray, scores: np.ndarray, gts: np.ndarray,
                iou_thr: float = IOU_MATCH):
    """贪心配对 → (tp, fp, fn)。规则：预测按分数从高到低，逐个去"认领"IoU 最大
    且还没被认领的真值框；IoU ≥ 阈值 → TP（找对），否则 FP（误报）；
    没被认领的真值 → FN（漏检）。"""
    used = np.zeros(len(gts), dtype=bool)
    tp = fp = 0
    order = np.argsort(-scores)               # 分数降序（argsort 给的是下标序）
    for i in order:
        ious = iou_matrix(preds[i][None], gts)[0].copy()
        ious[used] = -1.0                     # 已被认领的真值不再参选
        j = int(ious.argmax()) if len(ious) else -1
        if j >= 0 and ious[j] >= iou_thr:
            used[j] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~used).sum())


def predict(model, rec: dict) -> dict:
    """单图前向（eval + no_grad，系列同款）。
    注意：图【不】手动缩放——模型内部的 GeneralizedRCNNTransform 自己缩放，
    吐回来的框自动换算回原图坐标（1280×720 空间），和真值直接对得上。"""
    x = to_tensor(Image.open(rec["path"]).convert("RGB"))
    with torch.no_grad():
        return model([x])[0]


def save_vis(dir_path: Path, rec: dict, preds, scores, tag: str) -> None:
    """存一张对比图：绿框=真值（人标的），红框=模型检测（带置信度）。系列同款。"""
    dir_path.mkdir(exist_ok=True)
    img = Image.open(rec["path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    for g in rec["boxes"]:
        draw.rectangle([float(v) for v in g], outline=(0, 190, 0), width=2)
    for p, s in zip(preds, scores):
        draw.rectangle([float(v) for v in p], outline=(255, 0, 0), width=3)
        draw.text((p[0] + 2, max(0, p[1] - 22)), f"TL {s:.2f}", font=FONT, fill=(255, 0, 0))
    draw.text((6, 6), f"{tag}：绿=真值({len(rec['boxes'])}) 红=检测({len(preds)})",
              font=FONT, fill=(255, 255, 0))
    img.save(dir_path / (rec["path"].stem + ".jpg"), quality=90)


# ─────────────────────── [2] 考试主循环 ───────────────────────

def examine(model, records, save_dir):
    """逐图 前向 → 过滤(类别+分数) → 和真值配对。
    除 P/R/F1 外顺手记两项"惨状诊断"：
      ① 至少检出 1 个灯的图数（0 = 模型大面积"看不见"）
      ② 全场最高红绿灯分（如果连它都不过线 → 不是阈值卡死的，是真看不见）"""
    model.eval()
    tp = fp = fn = 0
    n_det = 0
    n_img_det = 0
    best = 0.0
    t0 = time.time()
    for i, rec in enumerate(records):
        out = predict(model, rec)
        # 两行掩码过滤：系列同款（类别=红绿灯 且 分数过线）
        mask = (out["labels"] == TL_ID) & (out["scores"] > SCORE_THR)
        preds = out["boxes"][mask].numpy()
        scores = out["scores"][mask].numpy()
        # 诊断②：不过线的红绿灯分也看一眼——模型"隐约看见但不敢说"就是这批
        tl_scores = out["scores"][out["labels"] == TL_ID]
        if len(tl_scores):
            best = max(best, float(tl_scores.max()))
        if len(scores):
            n_img_det += 1
        t, f, n = match_boxes(preds, scores, rec["boxes"])
        tp, fp, fn = tp + t, fp + f, fn + n
        n_det += len(preds)
        if i < N_SAVE:
            save_vis(save_dir, rec, preds, scores, "S2TLD考卷")
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            dt = (time.time() - t0) / (i + 1)
            eta = dt * (len(records) - i - 1) / 60
            print(f"    图 {i + 1:4d}/{len(records)}   累计 TP {tp}  FP {fp}  FN {fn}"
                  f"   {dt:.1f}s/图  ETA {eta:.0f}分")
    prec = tp / (tp + fp) if tp + fp else 0.0
    recl = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0.0
    return {"images": len(records), "detected_per_img": round(n_det / len(records), 2),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(recl, 3), "f1": round(f1, 3),
            "img_with_det": n_img_det, "best_score": round(best, 3)}


# ─────────────────────── 主流程 ───────────────────────

def main() -> None:
    print("═" * 62)
    print(" 📉 换考卷：S2TLD 交大小红绿灯 × 现有模型（尺子和旧考卷完全相同）")
    print(f" torch {torch.__version__} / torchvision {torchvision.__version__}（CPU）"
          f"{'  [冒烟模式]' if QUICK else ''}")
    print("═" * 62)

    # [0] 读考卷
    if not SAMPLES_JSON.exists():
        print(f"\n⚠ 找不到答案文件：{SAMPLES_JSON}")
        print("  先按 RUNBOOK 第 6 节下载 S2TLD 数据集（魔搭 ModelScope，国内直连）并解压。")
        sys.exit(1)
    print(f"\n[0] 读考卷：{SAMPLES_JSON.name} ...")
    img_root = find_image_root()
    records = load_s2tld(SAMPLES_JSON, img_root)
    if not records:
        print("⚠ 考卷里没有任何带红绿灯标注的图——文件下错了？")
        sys.exit(1)
    n_boxes = sum(len(r["boxes"]) for r in records)
    # np.concatenate([...])：把每张图的 (Ni,2) 沿第 0 轴摞成 (总框数, 2)（纯搬运族）
    #   实打实：[[21,52]] 和 [[9,31]] → [[21,52],[9,31]]
    # [:,2:] - [:,:2]：xyxy → 宽高（逐元素减）
    all_wh = np.concatenate([r["boxes"][:, 2:] - r["boxes"][:, :2] for r in records])
    avg_w, avg_h = all_wh.mean(0)
    # Counter：标准库计数器，list → {值: 次数}（一行顶手写 for 循环）
    # 状态 id 先过 S2TLD_STATES 翻译成人话再数
    dist = Counter(S2TLD_STATES.get(int(s), str(s))
                   for r in records for s in r["states"])
    print(f"    720x1280 子集：{len(records)} 张有灯图 / 共 {n_boxes} 个灯")
    print(f"    灯的个头：平均 {avg_w:.0f}×{avg_h:.0f} 像素（宽×高）——越小越难")
    coco = coco_tl_size()
    if coco:
        print(f"    对照旧考卷：COCO val 的灯平均 {coco[0]:.0f}×{coco[1]:.0f} 像素"
              f"（{coco[2]} 个）——个头差多少，难度就差多少")
    print("    状态分布：" + "、".join(f"{k} {v}" for k, v in dist.most_common(6)))

    # 均匀抽 EVAL_N 张：records[::step] 每隔 step 取 1 张（切片花活，纯搬运族）
    #   为什么不取前 N 张：samples 按拍摄顺序排列，前 150 张 ≈ 同几条街——考卷太窄
    step = max(1, len(records) // EVAL_N)
    picks = records[::step][:EVAL_N]
    print(f"    均匀抽 {len(picks)} 张参考试（每隔 {step} 张取 1 张）")

    # 图片文件存在性预检（fail fast：别等模型加载完才发现图不在）
    missing = [r["path"] for r in picks if not r["path"].exists()]
    if missing:
        print(f"\n⚠ 有 {len(missing)} 张考卷图片找不到，例如：")
        for p in missing[:3]:
            print(f"    {p}")
        print("    预期的摆放结构（详见 RUNBOOK 第 6 节）：")
        print("      minilops/data_cache/s2tld/dsdl_Det_full/…720x1280_samples.json")
        print("      minilops/data_cache/s2tld/S2TLD（720x1280）/normal_*/JPEGImages/*.jpg")
        print("    —— 脚本相对 samples.json 旁边的 s2tld/ 根目录去找图（自动兼容多包一层的情况）。")
        sys.exit(1)

    # [1] 模型（权重自动选择：和 demo_real_detect.py 同策略）
    print(f"\n[1] 加载 Faster R-CNN ...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    if CKPT.exists():
        # torch.load + load_state_dict + map_location：系列同款三件套（见 demo_real_detect）
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        print(f"    已叠加权重：{CKPT}")
    else:
        print(f"    没找到 {CKPT.name}，用官方 COCO 基线版")
    model.eval()
    with torch.no_grad():
        model([torch.zeros(3, 320, 320)])        # warm-up：系列惯例，第一枪慢，先空跑

    # [2] 考试
    print(f"\n[2] 考试：{len(picks)} 张 S2TLD 行车图，阈值 {SCORE_THR}，IoU 匹配 {IOU_MATCH} ...")
    m = examine(model, picks, BASE / "s2tld_eval")

    # [3] 成绩单（打印 + 落盘）
    n_gt = sum(len(r["boxes"]) for r in picks)
    print("\n" + "─" * 62)
    print(f" 📉 S2TLD 考卷成绩：考了 {m['images']} 图 / {n_gt} 个真值灯")
    print(f"    TP {m['tp']} | FP {m['fp']} | FN {m['fn']}"
          f"    →  P {m['precision']:.3f} | R {m['recall']:.3f} | F1 {m['f1']:.3f}")
    print(f"    惨状诊断：{m['img_with_det']}/{m['images']} 张图至少检出 1 个灯；"
          f"全场最高分 {m['best_score']}")
    if m["best_score"] < SCORE_THR:
        print(f"    ↑ 连最高分都没过阈值 {SCORE_THR}——不是阈值卡死的，是模型'看不见'这种小灯")
    log_path = BASE / "train_log.json"
    coco_log = None
    if log_path.exists():
        coco_log = json.load(open(log_path, encoding="utf-8"))
        print("\n    同一个模型，两张考卷：")
        print(f"      COCO val（预训练见过原题）  F1 {coco_log['baseline']['f1']:.2f}")
        print(f"      COCO val（COCO 微调后）      F1 {coco_log['after']['f1']:.2f}")
        print(f"      S2TLD  （从没见过的域）      F1 {m['f1']:.2f}   ← 这道鸿沟 = domain gap")

    result = {"config": {"eval_n": len(picks), "score_thr": SCORE_THR,
                         "iou_match": IOU_MATCH, "ckpt": str(CKPT),
                         "ckpt_used": CKPT.exists(), "quick": QUICK},
              "gt_stats": {"images_with_tl": len(records), "boxes": n_boxes,
                           "avg_w": round(float(avg_w), 1), "avg_h": round(float(avg_h), 1),
                           "states": dict(dist.most_common())},
              "s2tld": m,
              "coco_compare": ({"baseline_f1": coco_log["baseline"]["f1"],
                                "after_f1": coco_log["after"]["f1"]}
                               if coco_log else None)}
    json.dump(result, open(BASE / "s2tld_result.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    print("\n" + "═" * 62)
    print(" 成绩单落盘 → data_cache/s2tld_result.json（重看命令见 RUNBOOK 6.4）")
    print(" 肉眼看惨状 → data_cache/s2tld_eval/（绿=真值 红=检测）")
    print(" 惨是正常的——这一课的课题就叫 domain gap。")
    print(" 下一关：用 S2TLD 剩下的 4400+ 张图微调把分数救回来——")
    print(" 那才是数据闭环'采集→标注→再训练'的下半场（产线故事的完整闭环）。")
    print("═" * 62)


if __name__ == "__main__":
    main()
