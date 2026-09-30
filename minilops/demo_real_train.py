# -*- coding: utf-8 -*-
"""
🚦 真·训练闭环 —— 给 Faster R-CNN 接上 backward 和参数更新
==========================================================
demo_real_detect.py 只跑了链路的前一半（推理：eval 模式 + no_grad），
本文件补上后一半，把 retrain_feedback.py 的玩具闭环换成真货：

    [0] 读数据     mini_train / mini_val（COCO 格式 → 模型要的张量）
    [1] 基线评估   预训练权重先在 val 上考一次试（训练前成绩单）
    [2] 微调训练   train 模式 + loss.backward() + optimizer.step() ← 之前从没跑过的那半
    [3] 复评       同一张卷子再考一次（训练后成绩单）
    [4] 落盘       微调权重 .pth + loss 曲线和前后指标 train_log.json

五步心跳和玩具版（retrain_feedback.finetune / demo_traffic_light [3]）同一个骨架：

    玩具版（手写）                          真货版（本文件，torch 记账本代劳）
    ───────────────────────────           ─────────────────────────────────
    loss, gk, gb = core.loss_grad(x, t)   loss_dict = model([x], targets)
    core.kernel -= lr * gk                loss.backward()     ← 反向传播
                                           optimizer.step()   ← 参数更新

跑法（仓库根目录）：
    python -X utf8 minilops/demo_real_train.py            # 完整闭环（CPU 十来分钟）
    python -X utf8 minilops/demo_real_train.py --quick    # 冒烟：3 图 1 epoch，验证链路无报错

输出（都在 minilops/data_cache/ 下）：
    eval_before/  eval_after/   前 8 张 val 的可视化（绿框=真值，红框=检测），同名文件肉眼对比
    tl_finetuned.pth            微调后权重（以后可加载部署）
    train_log.json              每 step 的 loss + 前后指标（数据飞轮的证据）
    train_progress.txt          训练进度流水（长任务跑着时可随时打开看）
"""

import io
import sys
import json
import time
import random
from pathlib import Path

# Windows 控制台默认 GBK：包成 UTF-8 输出流（系列文件同款）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision

# ── COCO 类别表（和 demo_real_detect.py 同一张，复制进来保持两文件独立可跑）──
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
TL_ID = COCO_CLASSES.index("traffic light")   # = 10

BASE = Path(__file__).resolve().parent / "data_cache"
TRAIN_JSON, VAL_JSON = BASE / "mini_train.json", BASE / "mini_val.json"
PROGRESS_FILE = BASE / "train_progress.txt"

SCORE_THR = 0.5     # 分数阈值（产线旋钮；前后评估用同一把尺子才公平）
IOU_MATCH = 0.5     # 预测框和真值框 IoU ≥ 它才算"找对"（COCO 官方标准同款）

# ── 训练旋钮（产线超参雏形：每个值都值得问一句"为什么是这个"）──
LR = 2e-3               # 学习率：微调只能小步走，太大会冲掉预训练知识（灾难性遗忘）
EPOCHS = 2              # 完整训练集过几遍
MAX_TRAIN = 120         # 用前多少张训练图（CPU 时间预算；数据共 300 张）
FREEZE_BACKBONE = True  # 冻结卷积塔只训 RPN+ROI 头：反向快约一半 + 小数据不易过拟合
EVAL_N = None           # val 用多少张（None = 全部 60）
N_SAVE = 8              # 前后对比各存几张可视化
SEED = 0

QUICK = "--quick" in sys.argv   # 冒烟模式：只求全链路无报错，不求效果
if QUICK:
    EPOCHS, MAX_TRAIN, EVAL_N, N_SAVE = 1, 3, 4, 2


def load_font(size: int = 20):
    """Windows 自带粗体 Arial；找不到退回 PIL 默认小字体（系列文件同款）。"""
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = load_font()


# ─────────────────────── [0] 数据段 ───────────────────────
# ↑ 这一段就是上一关布置的 load_split 作业的参考答案——
#   回头把本函数遮住从零重写一遍、和它对拍结果，才算真正过关。

def load_split(json_path: Path) -> list:
    """COCO json → list[dict]，每张图一项：
       {"path": 图片路径, "w", "h",
        "boxes": (N,4) float32 xyxy,     ← 注意已经从 COCO 的 xywh 换算好
        "labels": (N,) int64}            ← category_id 原样保留（含全部类别，不只红绿灯）
    """
    data = json.load(open(json_path, encoding="utf-8"))
    # 表3 categories：id → 名字（只在打印统计时用）
    id2name = {c["id"]: c["name"] for c in data["categories"]}

    # 表2 annotations 按外键 image_id 归堆：{image_id: [annotation, ...]}
    #   dict.setdefault(key, [])：键不存在就先放个空列表——一行完成"分组"（数据库 GROUP BY 的手写版）
    anns_by_img = {}
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            continue          # crowd = 一大群物体挤成的一个框，训练噪声，扔
        anns_by_img.setdefault(a["image_id"], []).append(a)

    records = []
    for im in data["images"]:                 # 表1：一张图一行
        anns = anns_by_img.get(im["id"], [])
        if not anns:
            continue                          # 空标注图进训练会炸（targets 不能空），跳过
        # bbox xywh → xyxy：前两列（左上角）不动，后两列 += 前两列
        #   实打实：[149.39, 259.08, 11.12, 18.54] → [149.39, 259.08, 160.51, 277.62]
        #   列切片相加 = 上一关黑板上的公式（纯搬运族 + 逐元素加）
        boxes = np.array([a["bbox"] for a in anns], dtype=np.float32)
        boxes[:, 2:] += boxes[:, :2]
        records.append({
            "path": json_path.parent / im["coco_url"],   # coco_url 已改写成本地相对路径
            "w": im["width"], "h": im["height"],
            "boxes": boxes,
            "labels": np.array([a["category_id"] for a in anns], dtype=np.int64),
        })
    n_tl = sum(int((r["labels"] == TL_ID).sum()) for r in records)
    print(f"    {json_path.name}: {len(records)} 图 / "
          f"{sum(len(r['labels']) for r in records)} 框（红绿灯 {n_tl}）")
    return records


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL 图 → (3,H,W) float32 [0,1]。和 demo_real_detect.to_tensor 同款：
    np.array 复制 → permute 换轴（纯搬运族）→ .float()/255（逐元素）。"""
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def make_target(rec: dict) -> dict:
    """一条记录 → 模型训练要的 targets 格式（注意 boxes 是 xyxy float32，labels 是 int64）。
    ⚠ 类别号直通：COCO 的 category_id 和模型内部的类别下标是同一套编号
      （traffic light 两边都是 10）——当初建数据集保留全部 80 类就是为了这里不用重编号。"""
    return {"boxes": torch.from_numpy(rec["boxes"]),
            "labels": torch.from_numpy(rec["labels"])}


# ─────────────────────── 评估工具段（my_d3 将要写的 NMS 的表亲） ───────────────────────

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
    没被认领的真值 → FN（漏检）。——NMS 的表亲：NMS 用 IoU 压掉同类重叠框，
    这里用 IoU 认领真值。"""
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
    """单图前向（eval + no_grad，和 demo_real_detect.detect 同款）。"""
    x = to_tensor(Image.open(rec["path"]).convert("RGB"))
    with torch.no_grad():
        return model([x])[0]


def save_vis(dir_path: Path, rec: dict, preds, scores, gt, tag: str) -> None:
    """存一张对比图：绿框=真值（人标的），红框=模型检测（带置信度）。"""
    dir_path.mkdir(exist_ok=True)
    img = Image.open(rec["path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    for g in gt:
        draw.rectangle([float(v) for v in g], outline=(0, 190, 0), width=2)
    for p, s in zip(preds, scores):
        draw.rectangle([float(v) for v in p], outline=(255, 0, 0), width=3)
        draw.text((p[0] + 2, max(0, p[1] - 22)), f"TL {s:.2f}", font=FONT, fill=(255, 0, 0))
    draw.text((6, 6), f"{tag}：绿=真值({len(gt)}) 红=检测({len(preds)})",
              font=FONT, fill=(255, 255, 0))
    img.save(dir_path / rec["path"].name, quality=90)


def evaluate_tl(model, records, tag: str, save_dir=None):
    """红绿灯专项评估：60 张 val，每张图 前向 → 过滤(类别+分数) → 和真值配对。
    汇总 precision / recall / F1（retrain_feedback.evaluate 的真货版）。"""
    model.eval()
    tp = fp = fn = 0
    n_det = 0
    for i, rec in enumerate(records):
        out = predict(model, rec)
        # 两行掩码过滤：和 demo_real_detect.filter_traffic_lights 同款
        mask = (out["labels"] == TL_ID) & (out["scores"] > SCORE_THR)
        preds = out["boxes"][mask].numpy()
        scores = out["scores"][mask].numpy()
        gt = rec["boxes"][rec["labels"] == TL_ID]     # 真值里类别=10 的框
        t, f, n = match_boxes(preds, scores, gt)
        tp, fp, fn = tp + t, fp + f, fn + n
        n_det += len(preds)
        if save_dir is not None and i < N_SAVE:
            save_vis(save_dir, rec, preds, scores, gt, tag)
    prec = tp / (tp + fp) if tp + fp else 0.0
    recl = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0.0
    return {"images": len(records), "detected_per_img": round(n_det / len(records), 2),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(recl, 3), "f1": round(f1, 3)}


# ─────────────────────── [2] 训练段（本文件的心脏） ───────────────────────

def finetune(model, records, progress) -> list:
    """微调 = 五步心跳真货版。对照玩具版（见文件头注释）。
    SGD(lr, momentum=0.9)：带动量的下山——momentum 记住上一步的方向，
    走出"之"字时惯性帮你碾过去（demo_traffic_light 的 LR=8 纯裸降没这个待遇）。"""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=LR, momentum=0.9, weight_decay=1e-4)

    model.train()   # ← 训练模式：和 .eval() 相对的另一半世界。
    #   同一个 model 对象，两种人格：
    #     train 模式 → 吃 (图, targets)，吐 loss_dict（四项损失）
    #     eval  模式 → 只吃图，吐 boxes/labels/scores（检测框）
    #   忘了切模式 = 训练时拿到框、评估时拿到损失的"灵异 bug"

    order = list(range(len(records)))
    steps_log, step, t_start = [], 0, time.time()
    for ep in range(EPOCHS):
        random.shuffle(order)               # 打乱（DataLoader(shuffle=True) 的手写版）
        ep_loss = 0.0
        for idx in order:
            rec = records[idx]
            x = to_tensor(Image.open(rec["path"]).convert("RGB"))
            targets = [make_target(rec)]

            # ── 五步心跳 ──
            optimizer.zero_grad()           # ① 清上一步的梯度（不清会累加，经典坑）
            loss_dict = model([x], targets) # ② 前向：train 模式下返回的是四项损失
            #    loss_objectness  RPN 问"这有没有东西"
            #    loss_rpn_box_reg RPN 问"框画得准吗"
            #    loss_classifier  ROI 问"这东西是 80 类里哪一类"（红绿灯分数涨不涨看它）
            #    loss_box_reg     ROI 问"框精修得准吗"
            loss = sum(loss_dict.values())  # ③ 四项加总成总损失
            loss.backward()                 # ④ 反向：账本反着用，梯度挂到每个参数 .grad 上
            optimizer.step()                # ⑤ 参数 -= lr×梯度（替你写好了遍历）

            v = {k: lv.item() for k, lv in loss_dict.items()}   # .item()：张量→python数
            v["total"] = loss.item()   # .item()：张量→python 数并脱离记账本（float(loss) 连着账本会触发警告）
            v["step"] = step
            steps_log.append(v)
            ep_loss += v["total"]

            if step % 10 == 0 or step == EPOCHS * len(order) - 1:
                dt = (time.time() - t_start) / (step + 1)
                eta = dt * (EPOCHS * len(order) - step - 1) / 60
                line = (f"    step {step:4d}/{EPOCHS * len(order)}   "
                        f"loss {v['total']:.3f} (cls {v['loss_classifier']:.3f} "
                        f"box {v['loss_box_reg']:.3f})   "
                        f"{dt:.1f}s/步  ETA {eta:.0f}分")
                print(line)
                progress.write(line + "\n")
                progress.flush()
            step += 1
        print(f"  epoch {ep + 1}/{EPOCHS} 平均 loss {ep_loss / len(order):.3f}"
              f"   ← 判据看这里：epoch 均值的趋势才算数（batch=1 时单步 loss 乱跳是噪声）")
    return steps_log


# ─────────────────────── 主流程 ───────────────────────

def main() -> None:
    print("═" * 62)
    print(" 🚦 真·训练闭环：基线评估 → 微调(反向传播) → 复评 → 落盘")
    print(f" torch {torch.__version__} / torchvision {torchvision.__version__}（CPU）"
          f"{'  [冒烟模式]' if QUICK else ''}")
    print("═" * 62)

    random.seed(SEED)
    torch.manual_seed(SEED)

    # [0] 数据
    print("\n[0] 读数据（load_split = 上一关作业的参考答案）：")
    train_recs = load_split(TRAIN_JSON)[:MAX_TRAIN]
    val_recs = load_split(VAL_JSON)[:EVAL_N] if EVAL_N else load_split(VAL_JSON)
    print(f"    实际用：train {len(train_recs)} 图 × {EPOCHS} epoch | val {len(val_recs)} 图")

    # 模型（产线铁律：只加载一次，常驻内存；首次会下载 ~160MB 权重）
    print("\n[1] 加载 Faster R-CNN（COCO 预训练权重）...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    if FREEZE_BACKBONE:
        # 冻结 backbone：requires_grad=False → 反向传播路过它不更新、也不记它的账
        #   （转移学习惯例"楼不动只装修头"：卷积塔是通用视觉特征，2500 万参数
        #    不是 120 张图养得起的；RPN + ROI 头 ~1500 万参数才是要针对任务调的）
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        print(f"    backbone 已冻结（FREEZE_BACKBONE=True），只训 RPN + ROI 头")
    model.eval()
    with torch.no_grad():                    # warm-up：第一枪慢，空跑一发再计时（产线惯例）
        model([torch.zeros(3, 320, 320)])

    # [2] 基线评估（训练前成绩单）
    print(f"\n[2] 基线评估：预训练权重 × {len(val_recs)} 张 val，"
          f"阈值 {SCORE_THR}，IoU 匹配 {IOU_MATCH} ...")
    t0 = time.time()
    before = evaluate_tl(model, val_recs, "基线(训练前)", BASE / "eval_before")
    print(f"    基线：检出 {before['detected_per_img']:.2f} 框/图 | "
          f"P {before['precision']:.2f} | R {before['recall']:.2f} | F1 {before['f1']:.2f}"
          f"   (TP {before['tp']} / FP {before['fp']} / FN {before['fn']})"
          f"   [{time.time() - t0:.0f}s]")
    if before["fn"] > 0:
        print(f"    （FN={before['fn']} 个真值没找到——微调的目标就是让这个数变小）")

    # [3] 微调训练（反向传播 + 参数更新）
    print(f"\n[3] 微调训练：{len(train_recs)} 图 × {EPOCHS} epoch，"
          f"lr={LR}，五步心跳：zero→forward→loss→backward→step")
    progress = open(PROGRESS_FILE, "w", encoding="utf-8")
    steps_log = finetune(model, train_recs, progress)
    progress.close()

    # [4] 复评（同一张卷子）
    print(f"\n[4] 复评：同一批 {len(val_recs)} 张 val，同一把尺子 ...")
    after = evaluate_tl(model, val_recs, "微调(训练后)", BASE / "eval_after")
    print(f"    微调后：检出 {after['detected_per_img']:.2f} 框/图 | "
          f"P {after['precision']:.2f} | R {after['recall']:.2f} | F1 {after['f1']:.2f}"
          f"   (TP {after['tp']} / FP {after['fp']} / FN {after['fn']})")

    # [5] 落盘
    ckpt = BASE / "tl_finetuned.pth"
    torch.save(model.state_dict(), ckpt)
    log = {"config": {"lr": LR, "epochs": EPOCHS, "max_train": MAX_TRAIN,
                      "freeze_backbone": FREEZE_BACKBONE, "score_thr": SCORE_THR,
                      "iou_match": IOU_MATCH, "seed": SEED, "quick": QUICK},
           "baseline": before, "after": after, "steps": steps_log}
    json.dump(log, open(BASE / "train_log.json", "w", encoding="utf-8"), indent=1)

    # 成绩单对比
    print("\n" + "═" * 62)
    print(" 成绩单（val 同一批图，同一把尺子）：")
    print(f" {'':12s}{'基线(前)':>12s}{'微调(后)':>12s}{'变化':>10s}")
    for key, name in [("detected_per_img", "检出框/图"), ("precision", "precision"),
                      ("recall", "recall"), ("f1", "F1"), ("fn", "漏检 FN")]:
        d = after[key] - before[key]
        print(f" {name:12s}{before[key]:>12}{after[key]:>12}{d:>+10g}")
    print(f"\n 权重 → {ckpt.name} | 曲线和指标 → train_log.json")
    print(f" 肉眼对比：data_cache/eval_before/ vs eval_after/（同名文件，绿=真值 红=检测）")
    print("═" * 62)


if __name__ == "__main__":
    main()
