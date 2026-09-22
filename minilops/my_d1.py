import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        # 错误消息是算子契约的一部分（关7 的 400 响应要带原因）
        raise ValueError(f"to_grayscale expected (H, W, 3), got shape {img.shape}")

    # 权重取人眼敏感度（BT.601），和 CSS filter: grayscale(1) 同一张表
    weights = np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64) @ weights


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    H, W = img.shape                                  # 源图：H 行 W 列（老田）
    out = np.zeros((out_h, out_w), dtype=np.float64)  # 输出缓冲，先铺满 0（= new Float32Array）

    for i in range(out_h):                            # 每个输出行 = 撒一只豆子（y 方向）
        # ── ① 豆子落点，换到尺子B（黑板③+④压成一行）──
        #    (i + 0.5)    格心：格子号 + 半格（豆子在格中心，不在格边）
        #    * H / out_h  比例尺：新田每格 = 老田 H/out_h 块地宽
        #    - 0.5        换尺子B：从 0 号地中心量起（floor 直接给邻居、小数直接是权重）
        fy = (i + 0.5) * H / out_h - 0.5

        # ── ② 边缘加宽供样（黑板⑤）──
        #    不钳的话：豆子0 的 floor(-0.25) = -1，而 img[-1] 不报错、
        #    静默取最后一行 —— 图底混进图顶，无声无息
        fy = np.clip(fy, 0, H - 1)

        # ── ③ 上/下邻居 + 偏向权重 ──
        #    fy 只跟行号 i 有关 → 提到 j 循环外面：一整行共享一个 y 落点，算一次就够
        y0 = int(np.floor(fy))        # 上邻居行号（floor 往负无穷取整）
        y1 = min(y0 + 1, H - 1)       # 下邻居行号；最底行时钳住不许越界
        dy = fy - y0                  # 偏向下方的比例 = 纵向 lerp 权重
                                      # ⚠ 用 clamp 之后的 fy 算！坑就埋在这

        for j in range(out_w):        # 每个输出列（x 方向）
            # ── ①' x 方向镜像版：落点 → clamp → 邻居 ──
            #    ⚠ 必须用列号 j 算、且住在 j 循环里——每个列落点都不同
            fx = (j + 0.5) * W / out_w - 0.5    # 同款公式，横向来一遍
            fx = np.clip(fx, 0, W - 1)          # 同款救命一招：防 img[?, -1] 静默取尾列
            x0 = int(np.floor(fx))              # 左邻居列号
            x1 = min(x0 + 1, W - 1)             # 右邻居列号；最右列时钳住不许越界
            dx = fx - x0                        # 偏向右方的比例 = 横向 lerp 权重
                                                 # ⚠ 同样必须用 clamp 之后的 fx 算！

            # ── 四邻居：img[行, 列] —— 行在前！（新手经典滑倒点）──
            a = img[y0, x0]                     # 左上（y0 行 x0 列）
            b = img[y0, x1]                     # 右上
            c = img[y1, x0]                     # 左下
            d = img[y1, x1]                     # 右下

            # ── 两次 lerp：先横混两下，再竖混一下（黑板⑥）──
            top    = a * (1 - dx) + b * dx      # 上排：左右邻居按 dx 混
            bottom = c * (1 - dx) + d * dx      # 下排：左右邻居按 dx 混
            out[i, j] = top * (1 - dy) + bottom * dy   # 上下按 dy 混 → 写进输出格
    return out


def normalize(img: np.ndarray) -> np.ndarray:
    # 均值 0 方差 1；std 加 1e-8 防除零（全黑图 std=0 时炸 NaN）
    # img.mean()/img.std() 整批统计，无循环——numpy 的"整批操作"哲学
    return (img - img.mean()) / (img.std() + 1e-8)


if __name__ == "__main__":
    # ── 关1 自检（手册契约，全绿才算过）──
    img = np.random.default_rng(0).integers(0, 256, (37, 41, 3)).astype(np.uint8)
    out = normalize(resize_bilinear(to_grayscale(img), 16, 16))
    assert out.shape == (16, 16)
    assert abs(out.mean()) < 1e-9          # 归一化后均值≈0
    assert 0.2 < out.std() < 5
    print("[1/4] 管道 (37,41,3) -> (16,16) + 归一化: OK")

    # 契约：2D 输入必须在入口炸 ValueError，不能带病进模型
    try:
        to_grayscale(np.zeros((5, 5)))
        raise AssertionError("应报错")
    except ValueError:
        pass
    print("[2/4] to_grayscale 契约检查(2D输入报ValueError): OK")

    # resize 幂等：同尺寸返回原值
    g = np.arange(12).reshape(3, 4).astype(float)
    assert np.allclose(resize_bilinear(g, 3, 4), g)
    print("[3/4] resize 幂等(同尺寸返回原值): OK")

    # ── 黑板⑦的三个预言值：3×4 放大到 6×8 ──
    big = resize_bilinear(g, 6, 8)
    print(f"[4/4] 预言值: out[0][0]={big[0, 0]:.4f} (期望0.0000)"
          f"  out[0][1]={big[0, 1]:.4f} (期望0.2500)"
          f"  out[1][1]={big[1, 1]:.4f} (期望1.2500)")
    assert abs(big[0, 0] - 0.0) < 1e-9
    assert abs(big[0, 1] - 0.25) < 1e-9
    assert abs(big[1, 1] - 1.25) < 1e-9
    print("全部自检通过，关1 过关 ✅")
