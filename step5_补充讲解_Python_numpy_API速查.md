# Step 5 补充讲解：Python/numpy API 速查（按题目组织）

> 背景：概念你已经懂了（Step 1~4 打的底子），缺的是"词汇量"——看到 `np.clip` 不像看到 `clamp()` 那样零延迟反应。
> 用法：做题时撞见不认识的 API 就来这查；`★` 标记的是**心智模型级** API（不懂它 = 不懂那道题），必须内化成条件反射；其余是工具级，混个脸熟、用时再查。
> 所有对照都基于你已有的世界：JavaScript / WebGL / GLSL / shader。

---

## 一、心智模型级 ★（这 7 个必须条件反射）

### 1. `np.clip(x, lo, hi)` —— 就是 GLSL 的 `clamp()`
- 一维：`np.clip(-40.0, -35.2, 35.2)` → `-35.2`（钳到边界，不删元素）
- 撞见处：**A4**（Jitter 截断 ±0.02）、**A8**（PointClip 整题就是它）
- JS 版：`Math.min(Math.max(x, lo), hi)`
- 逐元素广播：传数组进去就对每个元素各自 clamp——numpy 的"整批操作"哲学，不用写 for

### 2. `np.floor(x)` —— 就是 GLSL 的 `floor()`
- `np.floor(3.7)` → `3.0`；负数注意：`np.floor(-3.2)` → `-4.0`（往负无穷取，不是往零取）
- 撞见处：**A11**（体素格坐标 `np.floor(coord / grid_size)`）
- 为什么是 floor 不是 round：格子要**左下角对齐**——round 会让格子边界落在 0.5 处，两个相邻点可能被 round 进同一格，破坏"格坐标 = 格身份"的唯一性

### 3. `np.cumsum(x)` —— 前缀和（prefix sum）
- `np.cumsum([120000, 98000, 150000])` → `[120000, 218000, 368000]`
- 撞见处：**B6**（offset 的生成——整题就是它）
- 你其实见过它：GPU 上 prefix sum 是经典并行算法（shared memory 树形归约）；`coord[:,0] = -coord[:,0]` 里的切片见第 5 条
- JS 版：`arr.reduce((acc, v) => [...acc, (acc.at(-1)||0)+v], [])`

### 4. `np.unique(x)` —— 数组去重
- 返回**排序后**的不重复元素；`len(np.unique(key))` = 不同值的个数
- 撞见处：**A11**（数格子数 = unique key 的长度）
- JS 版：`new Set(arr).size`——但 numpy 版自带排序，工业版 GridSample 用 `np.unique(..., return_index=True)` 同时拿到"每个格子的代表点下标"，一步到位
- `return_inverse=True`：额外返回"每个原始元素 → 哪个唯一值"的映射——**D6 铺回的 inverse 就是这个参数产的**

### 5. 花式索引 `a[idx]` —— gather！你 GPU 人的母语 ★★
- `idx` 是数组时：`seg[np.array([0,0,1,2,1])]` → `[seg[0], seg[0], seg[1], seg[2], seg[1]]`——**按下标表逐个取，允许重复**
- 撞见处：**D6**（`pred[inverse]` 一行完成铺回）、**A7**（SphereCrop 的 `coord[np.argsort(...)[:point_max]]`）
- **这就是 GPU 的 gather**：WebGL 里 index buffer 选顶点、textureLod 按 mip 索引采样，全是同一个操作——下标数组驱动数据搬运。反面是 scatter（`a[idx] = v`，按下标表写，会撞车）
- JS 版：`idx.map(i => seg[i])`——但 numpy 原生批量，无循环

### 6. 切片 `a[:, 0]` —— stride/offset 视图！
- `coord[:, 0]` = 所有行取第 0 列 = 取走 x 列；`coord[:, 0] = -coord[:, 0]` = 就地取反 x 列
- 撞见处：**A2**（RandomFlip 的翻转实现）
- 你早就懂它：这就是 WebGL vertex attribute 的 `stride`/`offset`——一块 (N,3) 的 buffer，按 stride=12 字节、offset=0/4/8 取出 x/y/z 通道。numpy 切片同样是**零拷贝视图**（view），不是复制
- JS 版：无（JS 没有多维数组，只能手写 `i*3+0`）——这是 numpy 对 JS 的降维打击之一

### 7. Set 运算 `&` `|` —— 交并
- `pred & target` → 交集；`pred | target` → 并集；Python 原生 set 也用同样符号
- 撞见处：**C6**（IoU = `len(pred & target) / len(pred | target)`——整题就是它）
- JS 版：`new Set([...a].filter(x => b.has(x)))`（交）——Python 的一行算符版优雅得多

---

## 二、随机数家族（做题撞见的都在这）

**统一入口：`rng = np.random.default_rng(42)`**——42 是种子，同种子必产同序列（可复现）。你熟悉的图形学版：shader 里的 hash 函数 / `srand(42)`，同一个动机——伪随机但确定性。

| API | 干什么 | 撞见处 |
|---|---|---|
| `rng.uniform(lo, hi)` | 均匀采样 [lo, hi) | A11（坐标）、A1（角度 = uniform(-1,1)×π） |
| `rng.uniform(0,1,size=(1000,3))` | 直接采一个 (1000,3) 数组——整批 | A11 |
| `np.random.rand() < 0.5` | 一次抛硬币 | A2/A3（Flip 的两次独立抛） |
| `rng.standard_normal((N,3))` | 标准正态（高斯）噪声 | A4（Jitter 的 sigma×randn） |
| `np.random.choice([0.5])` | 从列表里抽一个 | D2（TTA 固定角度的偷懒实现） |
| `np.random.random() > p` | 判断这次增强应用不应用 | A1 的 p=0.5 开关 |

和 torch 的对照（后面会越来越多见）：`torch.rand / torch.randn` 同款，加 `generator` 参数控种子。

---

## 三、argsort 家族（排序带下标回来）

```python
d = np.array([30, 10, 20])
np.argsort(d)        # → [1, 2, 0]  「谁排第几」——排序后各位置放的是原数组的下标
d[np.argsort(d)]     # → [10, 20, 30]  argsort + 花式索引 = 排序（gather 又出现了）
```
- 撞见处：**A7**（SphereCrop：`np.argsort(np.sum(np.square(coord-center), 1))[:point_max]` = "离球心最近的 point_max 个点的下标"）
- 拆开看：`np.square` 逐元素平方（无 sqrt，比距离省且单调等价）→ `np.sum(..., 1)` 沿 axis=1 求和（每行求和 = 每点到球心的平方距离）→ argsort 排出"由近到远"的下标表 → `[:point_max]` 切前 k 个 → 花式索引取走
- JS 版：`arr.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]).map(x=>x[1])`——numpy 三字母搞定

**`axis` 参数速记**：`axis=0` 沿行方向压（对每列操作）、`axis=1` 沿列方向压（对每行操作）。记不住就想：**axis= 要被消掉的那个维度**（呼应你学过的 einsum 读法："把指定维度求和消掉，其余保留"）。

---

## 四、Python 语言点（题面/仓库代码里高频）

| 语法 | 干什么 | 前端对照 |
|---|---|---|
| `d = dict(cfg)` / `cfg.pop("type")` | 浅拷贝 dict / 取走某个 key（删并返回） | `{...obj}` / `delete obj.k`（pop 有返回值） |
| `@decorator` | 装饰器：函数/类的包装器语法糖 | 高阶函数 `const wrapped = deco(fn)`；`@MODELS.register_module` ≈ `app.component()` |
| `getattr(obj, "name", None)` | 按字符串取属性，取不到给默认值 | `obj["name"] ?? null`——**按字符串**访问是钩子分发 `_trigger` 的核心 |
| `f"{x:.3f}"` | 格式化字符串 | `` `${x.toFixed(3)}` `` |
| `[f(x) for x in arr]` | 列表推导 | `arr.map(x => f(x))` |
| `{k: v for ...}` | 字典推导 | `Object.fromEntries(entries.map(...))` |
| `a if cond else b` | 三元表达式 | `cond ? a : b` |
| `**cfg` | dict 解包成关键字参数 `cls(**cfg)` | `fn(...args)` 的具名版：`fn(...Object.entries(cfg))` 的等价物 |
| `lambda x: x*2` | 匿名函数 | `x => x*2` |
| `weakref.proxy(self)` | 弱引用（不阻止 GC） | `WeakRef`——钩子持 trainer 弱引用防循环引用内存泄漏 |

---

## 五、torch 六兄弟（解析里出现的，先混脸熟）

```python
torch.cat([a, b], dim=1)      # 拼接（Collect 拼 feat 用 dim=-1：沿最后一维）
torch.softmax(x, dim=1)       # 沿 dim=1 归一化成概率分布
torch.randn(4, 32)            # 同 numpy 的 randn，但在 tensor 世界
model.eval() / model.train()  # 切模式：DropPath/Dropout 开关由它控制（B5）
tensor[..., 0:3]              # 切片同 numpy，... 表示"前面维度全要"
x.norm(dim=-1, keepdim=True)  # 沿最后一维求模长，keepdim 保留形状便于广播
```
- 心智模型照搬 numpy 即可（同一批人在设计 API）；新东西只有一个：`dim`/`axis` 的哲学 + tensor 带梯度记账（Step 4 讲过的"水管公式"层）

---

## 六、诊断：你缺的到底是哪一层

撞见陌生 API 时先分类，别一锅炖成"我不会 Python"：

1. **概念层**（cumsum=前缀和、gather=下标搬运、floor=格子对齐）——★ 级，必须内化。这些其实你 GPU 世界全有对应物，只是换了个马甲
2. **词汇层**（clip/cumsum/unique 这个函数名）——查表即得，见三次就认识了
3. **习惯层**（"整批操作别写 for"、"切片是视图不是拷贝"）——最慢但最值钱，写多了自然有

判断标准：**看懂这段代码在干嘛但自己写不出来** = 词汇层缺失（本速查解决）；**看不懂这段代码在干嘛** = 概念层缺失（回那道题的解析+📍位置块重看）。

---

## 七、三天内化路径（每天 10 分钟）

- **今天**：把第一节 7 个 ★ API 在 `python -X utf8` 交互环境里各敲一遍（复制下面的验证脚本）
- **明天**：重做 A11（手写 GridSample）不许看任何参考——三行能不能盲写出来是试金石
- **后天**：把 D6 的 inverse 铺回改写成"用 for 循环的笨版本"再对比一行花式索引——亲手感受 gather 的表达力

```python
# 验证脚本：一次性敲完 7 个 ★ API
import numpy as np
print(np.clip([-40, -30, 0, 34, 36.5], -35.2, 35.2))   # A8
print(np.floor(np.array([3.7, -3.2])))                  # 注意负数 → [ 3. -4.]
print(np.cumsum([120000, 98000, 150000]))               # B6 → [120000 218000 368000]
print(len(np.unique([1, 2, 2, 3, 3, 3])))               # → 4
seg = np.array([7, 9, 5]); inv = np.array([0, 0, 1, 2, 1])
print(seg[inv])                                          # D6 → [7 7 9 5 9]
coord = np.array([[1., 2., 3.], [4., 5., 6.]])
coord[:, 0] = -coord[:, 0]; print(coord)                 # A2：x 列取反
print(len({2,3,4} & {1,2,3}), len({2,3,4} | {1,2,3}))    # C6 → 2 4
```
