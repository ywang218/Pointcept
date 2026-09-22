import numpy as np

# ══════════════════════════════════════════════════════════
# 例题：1D 亮斑检测器（my_d2 的 1D 同构版，先读懂它再移植成 2D）
#
# 故事：线扫传感器吐出一条 1D 读数（暗处≈0.1，亮段≈0.9）。
# 模型：1×2 "卷积核"（w0, w1）+ bias，共 3 个参数。
# 语义：滑过每个窗口问同一句——"两个邻居都亮吗？"
#      都亮 → z 大 → sigmoid → p 接近 1 → 报告"这里有亮斑"
# ══════════════════════════════════════════════════════════


def make_pairs(x: np.ndarray) -> np.ndarray:
    """x:(N,) → (N-1, 2)  每个窗口位置的两邻居（im2col 的 1D 版）
    通道顺序：0↔左邻居(w0)，1↔右邻居(w1) —— ⚠ 必须和 kernel 顺序对上"""

    # ── 实打实例子：x = [10, 20, 30, 40, 50] ──
    #   x[:-1] = [10, 20, 30, 40]   （去掉最后一个）
    #   x[1:]  = [20, 30, 40, 50]   （去掉第一个）
    #   两者的第 i 个元素恰好是"窗口 i 的左邻居/右邻居"——错位一步的孪生切片
    #   （切片是零拷贝视图：不复制数据，只记"从哪到哪、步长多少"——stride/offset）
    #
    # np.stack = 配对打包机，不做任何算术。axis=-1 版本，实打实：
    #
    #   np.stack([x[:-1], x[1:]], axis=-1) =
    #       [[10, 20],   ← 窗口0 的 (左,右) = (x[0], x[1])
    #        [20, 30],   ← 窗口1 的 (左,右) = (x[1], x[2])
    #        [30, 40],   ← 窗口2
    #        [40, 50]]   ← 窗口3
    #   (N-1,) + (N-1,) --axis=-1--> (N-1, 2)：新增的"邻居身份"维放在最后
    #
    #   对照 axis=0 版（同一堆数换个摆法）：
    #       [[10, 20, 30, 40],    ← 整条"左邻居们"一行
    #        [20, 30, 40, 50]]    ← 整条"右邻居们"一行
    #   axis=-1 = 一对一对摆（配对优先）；axis=0 = 同类扎堆摆（分类优先）
    #
    # 选 axis=-1 的用意：让 (N-1,2) @ (2,) 的内维对上，
    #   一次矩阵乘算完全部窗口（下面 forward 里那一行）
    return np.stack([x[:-1], x[1:]], axis=-1)


def forward(kernel: np.ndarray, bias: float, x: np.ndarray) -> np.ndarray:
    """kernel:(2,) → p:(N-1,)  sigmoid 概率
    z = 每窗口 w·邻居 + bias（一个 matmul 全算完，无循环）"""
    pairs = make_pairs(x)
    # @ = 矩阵乘(np.matmul)：(N-1,2) @ (2,) → (N-1,)
    #   实打实（pairs=[[10,20],[20,30],[30,40],[40,50]], kernel=[0.4,0.3]）：
    #     窗口0: 10*0.4 + 20*0.3 = 10     ← 一个点积
    #     窗口1: 20*0.4 + 30*0.3 = 17     ← 又一个点积（同一组权重！= 参数共享）
    #     窗口2: 24.0    窗口3: 31.0       ← (N-1,2)@(2,) = 每行各做一次点积
    #   + bias：广播——一个标量加到所有 4 个 z 上（逐元素，无循环）
    z = pairs @ kernel + bias
    # np.exp(-z)：逐元素算 e^(-z_i)，形状不变；除法、加法同样逐元素
    #   实打实（z=0 时）：1/(1+e^0) = 1/2 = 0.5 —— z=0 → p=0.5，"拿不准"
    #   z=+2 → 1/(1+e^-2) ≈ 0.88（大正数→接近1）；z=-2 → ≈0.12（大负数→接近0）
    # sigmoid = 把任意实数压进 (0,1) 的"概率形状"函数
    return 1.0 / (1.0 + np.exp(-z))


def bce(p: np.ndarray, target: np.ndarray) -> float:
    """BCE：把每个窗口当二分类，对全图取 mean
    clip 是工业加固：p 精确等于 0/1 时 log 会炸 inf"""
    # np.clip(p, lo, hi) = GLSL 的 clamp()：把每个元素钳进 [lo,hi]，形状不变
    #   实打实：p=[0.5, 0.999...] → [0.5, 0.999999999999]（1e-12 那一侧钳住）
    p = np.clip(p, 1e-12, 1 - 1e-12)
    # np.log(p)：逐元素自然对数（e 为底）；乘法/加法也是逐元素
    #   实打实（p=0.61, t=1）：t*ln(p) + (1-t)*ln(1-p) = 1*ln(0.61) + 0*ln(0.39)
    #                        = ln(0.61) ≈ -0.49      ← 只有 t=1 那一支活着
    #   实打实（p=0.55, t=0）：= 0*ln(0.55) + 1*ln(0.45) ≈ -0.80 ← t=0 那支活着
    #   一个式子同时表达两个分支——switch 被算术吞掉了
    # np.mean(...)：全部元素求和÷个数 → 一个标量（BCE 的 /M 就在这一步）
    #   外面的负号：log 算出来是负的（ln(x)<0 当 x<1），取负变"罚分"——越大越坏
    return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))


def loss_grad(kernel, bias, x, target):
    """→ (loss, grad_kernel:(2,), grad_bias: float)
    作业 my_d2.loss_grad 的 1D 原型 —— backward 三行各有来历，见推导板书"""
    pairs = make_pairs(x)
    z = pairs @ kernel + bias
    p = 1.0 / (1.0 + np.exp(-z))
    loss = bce(p, target)

    # ── 反向传播三行，每行都是链式法则的一步 ──
    # ① dz = (p − target)/M：BCE∘sigmoid 对 z 的梯度（相消结论，板B有推导）
    #    实打实（p=0.552, t=0, M=11）：
    #      dz_i = (0.552 − 0) / 11 = 0.0502    ← "多报了 0.552，往回收一点"
    #    实打实（p=0.610, t=1, M=11）：
    #      dz_i = (0.610 − 1) / 11 = −0.0355   ← "少报了 0.39，再多报一点"
    #    逐元素减法，形状 (N-1,)；/p.size = 除以窗口总数（mean 求导的遗留）
    dz = (p - target) / p.size
    # ② pairs.T @ dz：(N-1,2) 转置成 (2,N-1) 再 @ (N-1,) → (2,)
    #    .T = 转置：行列互换的零拷贝视图
    #    数学上 = dL/dw_c = Σ_i dz_i · pairs[i,c]——w_c 出现在每个窗口，
    #    每个窗口贡献一份梯度，全加起来；矩阵乘一步求完这个和
    #    实打实（只有窗口0的 dz=1，其余 0；pairs[0]=[10,20]）：
    #      grad_w0 = 1*10 + 0*20 + 0*30 + ... = 10   ← 谁给 z 供了多少数，
    #      grad_w1 = 1*20 + ... = 20                   谁就分到多少责任
    grad_kernel = pairs.T @ dz
    # ③ dz.sum()：全部分量求和 → 标量（.sum() 求和族：shape 收缩成 0 维）
    #    bias 加进了每个窗口的 z（每份都是整份的 bias，不像 w 还乘了邻居值）
    #    → 梯度 = 全部 dz 直接加总，没有任何加权
    grad_bias = dz.sum()
    return loss, grad_kernel, grad_bias


def loss_of(params: np.ndarray, x: np.ndarray, target: np.ndarray) -> float:
    """params = [w0, w1, bias] 打包成一个向量 —— 只为数值梯度服务"""
    # params[:2] = 切片前两个 = (w0, w1)；params[2] = 第三个 = bias
    return bce(forward(params[:2], params[2], x), target)


def numerical_grad(params, x, target, eps=1e-6) -> np.ndarray:
    """数值梯度：(L(θ+ε) − L(θ−ε)) / 2ε
    完全不懂链式法则的"笨办法"——正因如此，它是审判手写 backward 的黄金标准"""
    # np.zeros(3)：分配一个 3 元素的全 0 数组（= new Float32Array(3)）
    g = np.zeros(3)
    for c in range(3):                  # 逐参数扰动：0,1=w0,w1；2=bias
        # .copy()：真复制一份（不是视图）——扰动不能污染原参数
        dp, dm = params.copy(), params.copy()
        dp[c] += eps                    # 只把第 c 个参数抬高 ε
        dm[c] -= eps                    # 只把第 c 个参数压低 ε
        # 中心差分：(高一点的loss − 低一点的loss) / 抬高量 ≈ 该参数处的斜率
        # 实打实（斜率 2、ε=1e-6）：(L(θ+ε)−L(θ−ε))/2ε = 2.000000… ≈ 2 ✓
        g[c] = (loss_of(dp, x, target) - loss_of(dm, x, target)) / (2 * eps)
    return g


if __name__ == "__main__":
    # ── 造一条带亮斑的 1D 信号 ──
    # np.random.default_rng(0)：带种子的随机数发生器——种子固定 → 每次跑出
    #   完全相同的"随机"数（可复现；rng.random(12) = 12 个 [0,1) 均匀随机数）
    rng = np.random.default_rng(0)
    x = 0.1 + 0.05 * rng.random(12)     # 暗噪声底：0.1~0.15 的随机扰动
    x[5:9] = 0.9 + 0.05 * rng.random(4) # 位置 5~8 亮段：0.9~0.95（切片赋值=逐段改写）
    # np.zeros(11)：11 个 0（11 = 12−1 个窗口）
    target = np.zeros(11)
    target[5:8] = 1.0                    # 两邻居都亮的窗口 = 5,6,7 三个位置

    # 未训练的随便一组参数（梯度审判不需要模型好）
    # np.array([...])：从列表造数组
    params = np.array([0.4, 0.3, -0.2])

    p = forward(params[:2], params[2], x)
    # f-string：{...} 里填变量值；:.3f = 保留3位小数
    # np.round(p[5:8], 3)：切片取 5~7 三个，逐元素四舍五入到 3 位小数
    print(f"信号 N=12 → 窗口数 11，p.shape = {p.shape}")
    print(f"亮斑窗口的 p 值: {np.round(p[5:8], 3)} （未训练，不用好）\n")

    loss, gk, gb = loss_grad(params[:2], params[2], x, target)
    # np.array([a, b, c])：把三个数打包回一个 (3,) 向量，好和数值梯度逐位比
    analytic = np.array([gk[0], gk[1], gb])           # 手写 backward 的答案
    num = numerical_grad(params, x, target)           # 笨办法的答案

    print(f"loss = {loss:.6f}\n")
    print(f"参数     手写梯度        数值梯度        差")
    # zip(名字表, 答案A, 答案B)：三张表按位置配对迭代；{a:+.8f} = 总带正负号8位小数
    for name, a, n in zip(["w0", "w1", "bias"], analytic, num):
        print(f"{name:5s} {a:+.8f}   {n:+.8f}   {abs(a - n):.2e}")

    # np.allclose(A, B, atol=1e-6)：逐位比 |A−B| < 1e-6 全成立才 True
    #   （不用 == 是因为浮点数直接比相等会输给舍入误差）
    assert np.allclose(analytic, num, atol=1e-6)
    print("\n数值梯度审判：3 个参数全对 ✅（手写 backward 正确）")
