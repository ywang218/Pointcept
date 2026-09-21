# step4 补充讲解：`.forward()` / `.backward()` 背后的执行机制

> 回答的问题：训练和推理都被封装成 `.forward()`/`.backward()`，计算损失就是 `loss()`？
> 这些细节算法工程师不需要知道吗？还是因为这部分是密集计算不适合 Python、封装好了调 GPU 的？
>
> 这是"反向传播"那课（数学层）的**工程补全**。读完这块，知识版图闭环：
> 数据链路 → 模型结构 → 训练机制 → 执行机制。

---

## 第一章：三种方法的真实身份——不是"藏细节"，是三种不同性质的东西

| 方法 | 它是什么 | 谁写的 | 在哪执行 |
|------|---------|--------|---------|
| `model.forward()` / `model(input)` | **你自己定义的计算流程**（哪些层、什么顺序连接） | 算法工程师（PTv2 的作者） | Python 只是"指挥"，**实际数值计算在 C++/CUDA** |
| `loss.backward()` | **自动微分引擎**——沿前向计算图反向算梯度 | PyTorch 框架作者 | 几乎全在 C++/CUDA |
| `optimizer.step()` | 更新规则（权重 -= lr×梯度，AdamW 加自适应花样） | PyTorch 框架作者 | C++/CUDA |
| `loss` 的计算 | 调 `criteria(logits, answer)`——也是一个 PyTorch 运算组合（CrossEntropyLoss 等），同样只是在 Python 层"点名" | 框架提供，config 里按 type 选用 | C++/CUDA |

关键认知：**`.forward()` 里的 Python 代码只是"指挥"，不是"演奏"。**

以已读过的 `GroupedVectorAttention.forward` 为例：

```python
query = self.linear_q(feat)          # Python: 一行
key = pointops.grouping(...)          # Python: 一行
weight = self.softmax(...)            # Python: 一行
feat = torch.einsum(...)              # Python: 一行
```

每一行 Python 实际发生的事：

```
Python 层:  "算一个矩阵乘法"           ← 你写的,纳秒级,只是发指令
    ↓
ATen 层:    C++ 张量库,拆解操作        ← 不可见的框架代码
    ↓
CUDA 层:    <<<grid, block>>> kernel   ← 真正的数亿级浮点运算,GPU 执行
```

**图形学生秒懂类比**：和 WebGL 写 shader 一模一样——

- 写 `gl.drawArrays(gl.TRIANGLES, 0, n)`，JS 只是发一条指令，真正的逐顶点/逐像素计算在 GPU 的 shader 里
- `forward()` 里每行 Python = JS 里一条 draw call；`linear_q(feat)` 内部启动的 CUDA kernel = vertex/fragment shader
- **你一直在用"JS 调 GPU"的方式做图形渲染，PyTorch 只是同一件事的数值计算版**

---

## 第二章：`backward()` 是真正的魔法——为什么 forward 你写、backward 永远没人写

对比看：

```python
# forward —— 你写(每个模型都不一样,这是算法创新所在):
y = relu(x @ W1 + b1)
z = y @ W2 + b2

# backward —— 你【永远不用写】(自动微分引擎自动生成):
# 给定 dz(上游误差),它自动知道:
#   dW2 = yᵀ · dz        (矩阵乘法对权重的导数)
#   dy  = dz · W2ᵀ       (传给下一层的误差)
#   dx  = dy · (x>0)     (relu 的导数:正数传,负数断)
#   dW1 = xᵀ · dy        ...一路链式法则下去
```

### 自动微分的原理（三步）

1. **记账**：你写 `forward` 时，PyTorch 在背后把每一步运算记在一张**计算图**（computational graph）上——`z` 是由 `y@W2` 算的，`y` 是由 `relu(...)` 算的……
2. **反向走图**：`loss.backward()` 沿这张图**从 loss 反向走到每个输入**，对每种运算查它的求导规则（C++ 实现的）
3. **链式连乘**：复杂函数的导数 = 基本运算导数的连乘（链式法则），每种基本运算的求导规则框架都预写好了——矩阵乘、加、relu、softmax、einsum……全有

**这就是"自动微分"四个字的全部含义。**

### 算法工程师要懂到什么程度？

不是"矩阵乘的梯度公式怎么推"，而是**懂流量的"水文学"，不用懂"水管公式"**：

- ✅ **梯度会流过哪些路径**——残差连接 `x + attn(x)` 让梯度有直通车道，所以深层网络训得动（这是设计 `Block` 结构时要懂的）
- ✅ **梯度为什么消失/爆炸**——sigmoid 导数最大 0.25，连乘十层梯度就没了；所以要换 relu、要 LayerNorm
- ✅ **什么时候 backward 会报错**——对没有梯度的 tensor 调 backward、计算图被 `no_grad` 截断等
- ❌ 每个算子的导数公式推导——框架管了
- ❌ kernel 怎么写、显存怎么排布——除非做 `libs/pointops` 那种自定义 CUDA 算子的细分专业岗

真出 bug 时（梯度爆炸 loss 变 NaN、显存泄漏），定位靠的就是上面那三个 ✅。

---

## 第三章：为什么 Python 能当"指挥"不能当"演奏"

直觉完全正确：**数值计算不适合 Python 执行**。两个原因：

### 1. 速度

Python 解释执行一个循环加法比 C 慢 100~1000 倍。12 万点 × 48 维的一个矩阵乘，Python for 循环写一次要几秒；CUDA 上是毫秒。

### 2. 并行

GPU 有一万个核心同时算；Python 有 GIL（全局解释器锁）连 CPU 多线程都跑不利索。

### 所以整个生态的分工

```
Python:   写"流程"(哪些层怎么连、loss 怎么算、多少 epoch)   ← 写起来爽,改起来快
C++/CUDA: 算"数值"(矩阵乘、归约、scatter)                 ← 快得飞起,写起来痛苦
          PyTorch 的 ~200 万行 C++ 代码就是这条边界上的翻译官
```

这正是 numpy 那课讲过的"整批操作"思想的延伸：numpy 是"Python 指挥 + C 演奏"（CPU 版），PyTorch 是"Python 指挥 + CUDA 演奏"（GPU 版）。**你在 step2 里 `np.floor(数组)` 一行处理 12 万个点时，就已经在用这个模式了**——那行 Python 也只是发了一条指令，真活是 numpy 的 C 代码干的。

---

## 第四章：算法工程师的认知分界线（总结表）

| 层 | 要不要懂 | 说明 |
|----|---------|------|
| 数值计算怎么调 GPU（kernel 编写、显存排布、warp 调度） | ❌ 不需要 | 细分专业岗（写 pointops 那种自定义算子的人）才需要 |
| 计算图/自动微分机制（梯度流、残差为什么 work、梯度消失） | ✅ **必须懂** | 模型设计和调 bug 的核心 |
| "Python 只是发指令、真活在 C++/CUDA" | ✅ **必须懂** | 不然永远看不懂为什么"一行 Python"要 50ms |
| numpy 的整批操作（vectorization） | ✅ 必须懂 | 同一思想的 CPU 版，已学（step2 逐行讲解） |

---

## 第五章：验证实验（一条命令，眼见为实）

### 实验 1：亲眼看到自动微分在干活

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -X utf8 -c "
import torch
# 计算图:requires_grad=True 让 PyTorch 开始记账
x = torch.tensor([2.0], requires_grad=True)
y = x * x * x          # forward: 8
y.backward()           # 眼不见为净的自动微分
print('y = x³ 在 x=2 处的导数:', x.grad)   # 应该是 3x² = 12
# 你没写任何求导代码 —— 链式法则被自动执行了
"
```

（装 CPU 版，约 200MB；别装 GPU 版，本机 4GB 显存 + 无 CUDA 环境）

### 实验 2（可选）：感受"Python 指挥 vs C 演奏"的速度差

```bash
python -X utf8 -c "
import numpy as np, time
n = 10_000_000
a = np.random.rand(n); b = np.random.rand(n)

t0 = time.perf_counter()
c = a + b                       # 一行 numpy:C 循环
t1 = time.perf_counter()
print(f'numpy 整批加法: {t1-t0:.4f}s')

t0 = time.perf_counter()
c = [a[i] + b[i] for i in range(n)]   # Python for 循环逐个加
t1 = time.perf_counter()
print(f'Python 循环加法: {t1-t0:.4f}s')
print('倍数:', )
"
```

同样的加法，Python 循环慢几十倍——这就是为什么整个深度学习生态是"Python 皮 + C 骨"。

---

## 第六章：与已学内容的连接（知识版图闭环）

| 模块 | 已学文档 | 本讲补的是什么 |
|------|---------|--------------|
| 数据链路 | 补充讲解 A（dataset/transform 两层） | —— |
| 模型结构 | 补充讲解 B 第 3 层（PTv2 三改动）+ C（QKV 纠正） | —— |
| 训练机制（数学层） | 补充讲解 B 第 1 层（链式法则/梯度/下山） | —— |
| **执行机制（工程层）** | **本讲** | 链式法则**由谁执行**（自动微分引擎）、**在哪执行**（C++/CUDA）、**为什么这样分工**（Python 慢且不能并行） |

四块拼起来，就是"算法工程师读任何一个模型代码"需要的全部地基。
