# step4 补充讲解 C：CUDA kernel 与自定义 shader 的对照

> 回答的问题：图形学有大量自定义 shader 代码，深度学习这里有体现吗？
> 答案：有，叫**自定义 CUDA kernel**——动机、执行模型、分层架构全部同构。

---

## 第一章：直接证据——仓库里就有真的"shader"

`libs/pointops/src/knn_query/knn_query_cuda_kernel.cu` 第 60 行：

```cuda
__global__ void knn_query_cuda_kernel(int m, int nsample,
        const float* __restrict__ xyz,      // 输入:所有点坐标 (n,3)
        const float* __restrict__ new_xyz,  // 查询点 (m,3)
        int* __restrict__ idx,              // 输出:每个点的K个邻居下标
        float* __restrict__ dist2) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;   // ← 就像 gl_GlobalInvocationID!
    if (pt_idx >= m) return;                              // ← 越界保护,和compute shader一模一样
    ...
}
```

### 逐概念对照表（同一门语言的两个方言）

| 图形学（WebGL/WebGPU/GLSL） | 深度学习（CUDA） | 点云仓库实例 |
|------|------|------|
| `#version 450` / `__KERNEL__` | `__global__ void kernel(...)` | knn_query_cuda_kernel.cu:60 |
| **顶点/片元**（每个顶点/像素跑一份） | **每个点跑一份**（每个线程处理一个点） | `pt_idx = blockIdx.x * blockDim.x + threadIdx.x` |
| `gl_GlobalInvocationID`（WebGPU compute） | `blockIdx * blockDim + threadIdx` | 同上，数学完全一样 |
| `buffer` / SSBO 传入顶点数据 | 指针参数传入张量数据 | `const float* xyz` |
| `out vec4 fragColor` | 输出指针 `int* idx` | 写邻居下标 |
| `__restrict__`（编译器优化提示） | `__restrict__`（完全相同） | 到处都是 |
| `dispatch(组数, 1, 1)` | `kernel<<<grid, block>>>` 启动 | 在 .cpp 绑定层 |
| **手写 GLSL 因为 JS 慢千倍** | **手写 CUDA 因为 Python 慢千倍** | **同一个动机** |

---

## 第二章：为什么要手写——标准算子 = 固定管线，不够用

PyTorch 的 200 万行 C++ 为什么不够 Pointcept 用？与图形学完全同构：

**标准算子 = 图形 API 内置固定管线**（blinn-phong、固定混合）——通用够快，但表达不了特殊数据访问模式。

`knn_query` 的例子：对每个查询点，在无序点数组里找最近 16 个。用 PyTorch 标准算子组合（广播算全距离矩阵再 topk）：
- 显存：N×M 距离矩阵，12万×12万 = 144 亿浮点 = **57GB**——放不下
- 必须：逐点局部计算 + 线程内维护小顶堆

源码 15~42 行的 `reheap`/`heap_sort`：**每个线程用一个大小为 K 的小顶堆维护"目前为止最近的 16 个邻居"**——教科书算法搬进 GPU。

### 这个仓库的"shader 库"（libs/pointops/src/）

- `knn_query`：K 近邻查询
- `ball_query`：球邻域查询（雷达点云传统）
- `grouping`：按邻居表 gather 特征（= instanced attribute gather）
- `interpolation`：三线性插值上采样
- `subtraction/aggregation`：特征差/聚合

---

## 第三章：深度学习的"渲染管线"分层对照

```
图形学                        深度学习
─────────────────────       ─────────────────────
JS/TS 应用层                 Python 应用层(model.forward里的每行)
        ↓                         ↓
Three.js(场景图/工具库)        PyTorch nn.Module(层抽象/autograd记账)
        ↓                         ↓
WebGL/WebGPU API              ATen C++(张量库,~200个标准算子)
        ↓                         ↓
【自定义 GLSL/WGSL shader】   【自定义 CUDA kernel ← libs/pointops】
        ↓                         ↓
GPU 硬件                      GPU 硬件(同一块卡!同一种SIMT执行模型)
```

**最妙的一点**：CUDA 和现代 GPU 图形的执行模型是同一个——**SIMT（单指令多线程）**，就是 compute shader。WebGPU compute shader 的心智模型（workgroup、每个 invocation 处理一个元素、shared memory 做 tile 内通信、barrier 同步）**原封不动**适用于 CUDA kernel。**这不是类比，是同一个硬件架构的两种编程接口。**

---

## 第四章：算法工程师要会写 kernel 吗？——分岗位

| 角色 | kernel 要求 |
|------|------------|
| 纯模型研究（改结构、发论文） | 基本不用写——优先用标准算子组合；kernel 是"别人的轮子" |
| 造轮子/框架开发（写 pointops 这类） | 必须精通（CUDA 专业岗） |
| **工程落地/算子封装（领导给的方向）** | **看懂 + 会改**：能读懂 kernel 逻辑、判断瓶颈在哪、知道何时"标准算子组合不出来"——这个层次图形背景几乎免费拥有 |

---

## 第五章：PyTorch 的"逃逸舱口"梯度

写自定义 kernel 不一定要写裸 CUDA——渐进层级（类似"先用 Three.js，不够再写 shader，再不够写 WebGPU 原生"）：

1. **标准算子组合**（95% 的情况够用）——Python 层拼
2. **torch.compile / Triton**——Python 语法写 kernel，编译器生成 CUDA（≈ shader 框架层）。Triton 当代越来越主流（FlashAttention 就是 Triton 写的）
3. **裸 CUDA 扩展**（pointops 走的路）——极致性能 + 最难写

梯度类比：内置材质 → 节点材质编辑器 → 手写 GLSL。

---

## 第六章：给学习者的落地建议

先不动手写，但做一次**"读 kernel 练习"**：

- 文件：`libs/pointops/src/knn_query/knn_query_cuda_kernel.cu`（~120 行，一半是堆排序工具）
- 核心 kernel 只有 60~100 行
- 读完会发现：除了语法（`__global__`/`__device__`），每行的语义都在 compute shader 里写过等价物
- 意义：消除"CUDA 恐惧"——将来遇到性能问题，能判断"这是 Python 层的问题还是 kernel 层的问题"

**总结**：图形学的自定义 shader 在深度学习里叫自定义 CUDA kernel。动机相同（宿主语言慢千倍 + 内置管线表达不了特殊数据访问模式）、执行模型相同（SIMT/compute shader 同源）、分层架构相同（应用层→框架层→API层→shader/kernel层→同一块 GPU）。点云恰好是 kernel 密集领域（无结构数据的 gather/scatter/邻域查询，标准算子天然不友好）——**shader 经验在这个领域是直接可迁移的生产力**。
