# -*- coding: utf-8 -*-
"""
🚦 真·红绿灯检测 —— COCO 预训练 Faster R-CNN 跑真实照片
=========================================================
和 demo_traffic_light.py（5 参数玩具）是【同一条骨架】：

    照片 → 预处理 → 模型前向 → 过滤/解码 → 画框保存

区别只是每一格从玩具换成真货：

    玩具版                            真货版（本文件）
    ──────────────────────────       ──────────────────────────────────
    自己合成的图                      真实照片（PIL 读取）
    手写双线性 resize（my_d1）        PIL 内置 resize（同一个算法）
    2×2 卷积核：5 个参数              Faster R-CNN：约 4 千万个参数
    1 张图、手写梯度下降训 200 步      COCO 数据集（33 万张图）上训好的权重
    手写 decode + NMS（my_d3 待写）   NMS 在模型内部；外面只剩 分数+类别 过滤

跑法（在仓库根目录下）：
    python -X utf8 minilops/demo_real_detect.py                # 处理 real_photos/ 全部图片
    python -X utf8 minilops/demo_real_detect.py 你的照片.jpg   # 或指定图片
    python -X utf8 minilops/demo_real_detect.py --ckpt 路径.pth 图片.jpg
                                    # ↑ 用微调权重（默认自动找 data_cache/tl_finetuned.pth）

输出：控制台检测表 + 每张图旁边生成 xxx_detected.jpg（画好框的副本）。
首次运行会下载权重（~160MB，一次性，之后走本地缓存）。
照片来自 Wikimedia Commons（来源见 real_photos/CREDITS.txt）。
"""

import io
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK：包成 UTF-8 输出流，中文/emoji 才不乱码（demo_traffic_light 同款）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── torch 登场前的说明（按约定：用库之前先讲它在干嘛）──
# torch = numpy 的"超级替身"：同样的多维数组（这里叫 tensor/张量），外加三件 numpy 没有的：
#   ① 自动微分：记账本，训练用。本 demo 是推理 → 用 no_grad 把记账关掉（见 detect()）
#   ② GPU 执行：本机装的是 CPU 版（推理够用；GPU 轮子以后 E 系列再说）
#   ③ model zoo：官方预训练模型库——本文件的主角 Faster R-CNN 就是从这拿的，
#      权重是别人在 COCO（33 万张图、150 万个标注框、80 类物体）上训好的
import torch
import torchvision

# ─────────────────────── 参考数据：COCO 类别表 ───────────────────────
# COCO 官方 91 类名（含背景，下标 = 类别号；'N/A' 是 COCO 编号史留的洞，无视）
# 这是查阅表不是教学内容。唯一要记住的事：traffic light 的类别号
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
# 用 index() 现算类别号，不靠手数（手数类别号 = 经典翻车点）
TL_ID = COCO_CLASSES.index("traffic light")   # = 10

SCORE_THR = 0.5          # 分数阈值：低于它的框不要（产线旋钮，调低=召回多误报也多）
PHOTO_DIR = Path(__file__).resolve().parent / "real_photos"
# 微调权重（demo_real_train.py 的产物）。用法：
#   python -X utf8 minilops/demo_real_detect.py --ckpt .../tl_finetuned.pth [图片]
#   不给 --ckpt → 默认找 data_cache/tl_finetuned.pth，有就自动用（微调版）
#   没有这个文件 → 退回官方 COCO 预训练权重（基线版）
CKPT = Path(__file__).resolve().parent / "data_cache" / "tl_finetuned.pth"
for i, a in enumerate(sys.argv):        # 手写参数解析：--ckpt 的下一个参数是权重路径
    if a == "--ckpt" and i + 1 < len(sys.argv):
        CKPT = Path(sys.argv[i + 1])
        del sys.argv[i:i + 2]           # 吃掉这两个参数，剩下的才当图片路径
        break


def load_font(size: int = 20):
    """Windows 自带粗体 Arial；找不到就退回 PIL 默认小字体。"""
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = load_font()


# ─────────────────────── [1] 预处理段（对应关1 的真货版） ───────────────────────

def load_photo(path: Path, max_side: int = 800) -> Image.Image:
    """读照片 + 缩到最长边 800（CPU 推理友好；PIL 的 BILINEAR 就是手写那款算法）。"""
    # Image.open：惰性读取（真到用数据才解码）；.convert("RGB")：任何格式→3通道 RGB
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        # Image.Resampling.BILINEAR：四邻居 lerp 混合——my_d1 里手写的那 20 行，
        # 工业库一行调用完事；round() 四舍五入取整像素尺寸
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL 图 → torch 张量 (3,H,W) float32 [0,1] —— 模型要的输入格式。
    对应关1 预处理：布局转换 HWC→CHW + 除 255 归一化（这个模型不吃 mean/std）。"""
    # np.array(img)：PIL→numpy，(H,W,3) uint8（np.array 一定复制数据，不像 asarray 可能共享）
    #   实打实：红色交通灯像素 → [235, 60, 60]
    arr = np.array(img)
    # torch.from_numpy(arr)：numpy→torch 张量，共享同一块内存（零拷贝，和切片视图同思想）
    #   ——从这行起进入 torch 世界，后面所有运算 torch 都会记账（自动微分），
    #   推理时我们用 no_grad 明说"只前向，不用记账"（见 detect()）
    x = torch.from_numpy(arr)
    # .permute(2,0,1)：维度重排 (H,W,3)→(3,H,W)（零拷贝换轴，纯搬运族）
    #   为什么：torch 视觉模型的约定是"通道在前"——WebGL 纹理反过来，认通道优先
    #   实打实：(480, 640, 3) → (3, 480, 640)
    # .float()：uint8→float32；/255.0：逐元素除，[0,255]→[0,1]
    #   实打实：[235, 60, 60] → [0.922, 0.235, 0.235]
    return x.permute(2, 0, 1).float() / 255.0


# ─────────────────────── [2] 模型段（加载一次 + warm-up） ───────────────────────

def load_model() -> torch.nn.Module:
    """加载模型。产线铁律：模型只加载一次，常驻内存。

    两档权重：
      data_cache/tl_finetuned.pth 存在 → 微调版（demo_real_train.py 训的，红绿灯专项加强）
      不存在                        → 官方 COCO 预训练基线版

    模型内部地图（拆解学习时按这个顺序挖）：
      backbone  ResNet50 卷积塔：浅层学边缘/颜色，深层学形状/物体部件
                ——就是你那个 2×2 卷积核长成 50 层大楼的样子
      fpn       特征金字塔：同一张图出多档分辨率的特征图（≈mipmap/LOD！），
                大目标看粗档、小目标看细档——近大远小各取所需
      rpn       区域建议网：在每档特征上问"这格附近有没有东西"
                ——就是你那个热力图思想的放大版（有目标的位置分数高）
      roi_heads 对每个候选框抠特征 → 分类成 80 类之一 + 精修框坐标；
                NMS 也藏在这一段里（模型吐出来的框已经是去过重的）
    """
    # weights="DEFAULT"：先拿官方 COCO 训练版——既是基线，也是微调的骨架
    print("[1] 加载 Faster R-CNN（COCO 预训练权重，首次下载 ~160MB）...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    if CKPT.exists():
        # torch.load(CKPT)：把训好的参数字典从盘上读回内存（纯搬运族，无计算）
        # load_state_dict：逐参数对号入座——结构和训练时一模一样才能坐得进去
        # map_location="cpu"：训在哪存哪无所谓，读回来强制放 CPU（本机没 GPU）
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        print(f"    已叠加微调权重：{CKPT}")
    else:
        print(f"    未找到微调权重（{CKPT.name}），用基线版。")
        print(f"    想用微调版：先跑 demo_real_train.py，或 --ckpt 指定路径")
    # .eval()：推理模式——关掉训练专用行为（BN 层用固定统计值、Dropout 关闭）
    #   忘了这行是经典 bug：模型行为漂移、结果每次不一样
    model.eval()
    return model


def warm_up(model) -> None:
    """空跑一发。第一枪总是慢（懒初始化/内存分配），产线惯例先空跑再接客——
    E 系列 warm-up 实验的同款动作。"""
    dummy = torch.zeros(3, 320, 320)   # 全 0 假图，只为触发内部初始化
    with torch.no_grad():
        model([dummy])


# ─────────────────────── [3] 前向 + 过滤段（decode 的真货版） ───────────────────────

def detect(model, img: Image.Image):
    """前向一次，返回模型原始输出（含全图所有类别的框）。"""
    x = to_tensor(img)
    t0 = time.time()
    # torch.no_grad()：上下文管理器——里面的运算不记账（不存中间结果给反传用）
    #   推理永远开它：省内存、快一截。训练才需要记账
    with torch.no_grad():
        # model([x])：输入是"图的列表"（这模型天生吃一个 batch）；输出同长度的列表
        #   [0] 取出批里第一张（也就这一张）的结果——一个 dict：
        #     boxes  (N,4)  每个检测框 (x0,y0,x1,y1)，像素坐标
        #     labels (N,)   每个框的 COCO 类别号
        #     scores  (N,)  每个框的自信分 [0,1]
        #   ⚠ NMS 已在模型内部做完——外面不用再去了重
        out = model([x])[0]
    return out, time.time() - t0


def filter_traffic_lights(out: dict):
    """从原始输出里挑出'分数够高 且 类别=红绿灯'的框——这就是 decode 的真货版
    （my_d3 待写的 decode+NMS，在这里被压缩成两行掩码操作）。"""
    scores, labels = out["scores"], out["labels"]
    # (labels == TL_ID)：张量逐元素比较 → 布尔掩码
    #   实打实：labels=[10, 3, 10] == 10 → [True, False, True]
    # (scores > SCORE_THR)：同款逐元素比较
    # & ：两个布尔掩码逐元素"与"——两个条件都要满足
    #   实打实：[T,F,T] & [T,T,F] → [T,F,F]
    mask = (labels == TL_ID) & (scores > SCORE_THR)
    # out["boxes"][mask]：布尔掩码当索引 = 只挑 True 的行（花式索引，纯搬运族）
    return out["boxes"][mask], scores[mask]


# ─────────────────────── [4] 可视化段 ───────────────────────

def annotate(img: Image.Image, out: dict, save_path: Path) -> int:
    """画框存图：红绿灯=红粗框，其他目标=灰细框（顺便看模型眼里整条街都有什么）。"""
    draw = ImageDraw.Draw(img)
    scores, labels, boxes = out["scores"], out["labels"], out["boxes"]
    n_saved = 0
    for label, score, box in zip(labels.tolist(), scores.tolist(), boxes):
        # box.tolist()：张量行→python 浮点列表
        #   实打实：tensor([451.3, 22.7, 502.1, 148.9]) → [451.3, 22.7, 502.1, 148.9]
        x0, y0, x1, y1 = box.tolist()
        if int(label) == TL_ID and score > SCORE_THR:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
            # max(0, y0-24)：标签放框上方，框贴图顶时钳回 0（GLSL clamp 思想）
            draw.text((x0 + 2, max(0, y0 - 24)), f"traffic light {score:.2f}",
                      font=FONT, fill=(255, 0, 0))
            n_saved += 1
        elif score > SCORE_THR:
            draw.rectangle([x0, y0, x1, y1], outline=(170, 170, 170), width=2)
            draw.text((x0 + 2, max(0, y0 - 22)), f"{COCO_CLASSES[label]} {score:.2f}",
                      font=FONT, fill=(170, 170, 170))
    # img.save：PIL 编码写盘；quality=90 是 JPEG 压缩档位
    img.save(save_path, quality=90)
    return n_saved


# ─────────────────────── 主流程 ───────────────────────

def main() -> None:
    print("═" * 62)
    print(" 🚦 真·红绿灯检测：预训练 Faster R-CNN × 真实照片")
    print(f" torch {torch.__version__} / torchvision {torchvision.__version__}"
          f"（CPU 推理，几秒一张）")
    print("═" * 62)

    # 命令行给了图片就用给的；没给就处理 real_photos/ 全部
    photos = [Path(a) for a in sys.argv[1:]]
    if not photos:
        photos = sorted([*PHOTO_DIR.glob("*.jpg"), *PHOTO_DIR.glob("*.jpeg"),
                         *PHOTO_DIR.glob("*.png")])
    if not photos:
        print(f"\n没找到图片。把照片放进 {PHOTO_DIR} 后重跑，")
        print("或：python -X utf8 minilops/demo_real_detect.py 你的照片.jpg")
        sys.exit(1)

    model = load_model()
    print("[2] warm-up 空跑一发（产线惯例：第一枪慢，先空跑再接客）...")
    warm_up(model)

    total_hit = 0
    for photo in photos:
        img = load_photo(photo)
        print(f"\n── {photo.name}（{img.width}×{img.height}）──")

        out, dt = detect(model, img)
        print(f"[3] 模型前向 {dt:.2f}s，吐出 {len(out['scores'])} 个框（内部已 NMS）")

        # 检测表：全部类别按分数排，看模型眼里的整条街
        # zip + sorted：打包 (名称,分数,框) 再按分数降序；lambda s: -s[1] = 按分数从高到低
        rows = [(COCO_CLASSES[int(l)], float(s), b.tolist())
                for l, s, b in zip(out["labels"], out["scores"], out["boxes"])]
        rows.sort(key=lambda r: -r[1])
        print("    类别             分数   框(x0, y0, x1, y1)")
        for name, score, box in rows[:10]:
            bi = [round(v) for v in box]        # round：坐标四舍五入好读
            print(f"    {name:15s} {score:.2f}   {bi}")
        if len(rows) > 10:
            print(f"    ...（其余 {len(rows) - 10} 个略）")

        # 红绿灯专项过滤
        tl_boxes, tl_scores = filter_traffic_lights(out)
        if len(tl_scores) == 0:
            # 找全场最高分的红绿灯（哪怕没过线）——告诉你阈值离它多远
            tl_all = [(float(s), b) for l, s, b in zip(out["labels"], out["scores"], out["boxes"])
                      if int(l) == TL_ID]
            if tl_all:
                best = max(tl_all, key=lambda t: t[0])
                print(f"    ⚠ 没有红绿灯过 {SCORE_THR} 阈值；最高分仅 {best[0]:.2f}"
                      f"（阈值是产线旋钮：调低→召回多、误报也多）")
            else:
                print("    ⚠ 模型认为图里没有红绿灯")
        else:
            for s, b in zip(tl_scores.tolist(), tl_boxes.tolist()):
                print(f"    🚦 检测到红绿灯：置信度 {s:.2f}，框 "
                      f"{[round(v) for v in b]}")
            total_hit += len(tl_scores)

        # 画框保存
        save_path = photo.with_name(photo.stem + "_detected.jpg")
        n = annotate(img, out, save_path)
        print(f"[4] 画框存图：{save_path}（红框 {n} 个）")

    print("\n" + "═" * 62)
    print(f" 全部完成：{len(photos)} 张照片，共 {total_hit} 个红绿灯框。")
    print(" 打开 *_detected.jpg 亲眼看效果。哪一块想拆开（backbone/FPN/RPN/ROI）")
    print(" ——指给我，一块块抠。")
    print("═" * 62)


if __name__ == "__main__":
    main()
