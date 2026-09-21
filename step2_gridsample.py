"""
Step 2: 亲手实现 GridSample 体素化 —— 今晚最核心的实验

对应仓库代码: pointcept/datasets/transform.py:840 的 GridSample 类
对应模型概念: PTv2 的 grid_sizes=(0.06, 0.12, 0.24, 0.48) 四级下采样

运行: python -X utf8 step2_gridsample.py

重要背景（为什么以前你觉得"点云没有重复点"）:
  单帧随机合成点太稀疏,GridSample 压不动。真实的压缩来源有两个:
  1. 表面采样 —— 真实世界点都贴在地表/墙面/物体表面(2D 流形嵌在 3D 里)
  2. 多帧堆叠 —— SemanticKITTI 常用 10 帧叠加,同一表面被反复扫到
  本脚本的实验二/三就用这两种真实结构。
"""
import numpy as np
import open3d as o3d

rng = np.random.default_rng(42)

# ============================================================
# 实验一：稀疏单帧(对照组) —— 随机撒点,几乎压不动
# ============================================================
print("=" * 62)
print("实验一(对照组): 随机稀疏点,GridSample 几乎不压缩")
print("=" * 62)

n = 120000
angles = rng.uniform(0, 2 * np.pi, n)
r = np.sqrt(rng.uniform(0, 1, n)) * 35.0
z = rng.uniform(-2.0, 3.0, n)
coord_spray = np.stack([r * np.cos(angles), r * np.sin(angles), z], axis=1)
print(f"\n原始: {n} 点(70m x 70m 体积内随机撒,点距远大于格子)\n")

for gs in [0.05, 0.06, 0.12, 0.24, 0.48]:
    grid = np.floor(coord_spray / gs).astype(np.int64)
    key = grid[:, 0] * 10**10 + grid[:, 1] * 10**5 + grid[:, 2]
    nv = len(np.unique(key))
    tag = {0.06: " <- PTv2 enc 第1级", 0.12: " <- PTv2 enc 第2级",
           0.24: " <- PTv2 enc 第3级", 0.48: " <- PTv2 enc 第4级"}.get(gs, "")
    print(f"  grid_size={gs:<5}: {nv:>7} 体素 (压缩 {n/nv:.1f}x){tag}")

print("\n  => 结论: 体积内随机点太稀,格子总是'一格一点'。真实压缩靠表面采样 ↓")

# ============================================================
# 实验二：表面采样(真实结构) —— 地面+墙+车,压缩率立刻上来
# ============================================================
print("\n" + "=" * 62)
print("实验二: 表面采样场景(地面 8 万 + 墙 3 万 + 车 1 万)")
print("=" * 62)

# 地面: 50m x 50m,带微小传感器噪声
gx = rng.uniform(-25, 25, 80000)
gy = rng.uniform(-25, 25, 80000)
gz = rng.normal(0, 0.005, 80000)
ground = np.stack([gx, gy, gz], axis=1)

# 墙: x=25 处的竖直平面
wx = np.full(30000, 25.0) + rng.normal(0, 0.005, 30000)
wy = rng.uniform(-25, 25, 30000)
wz = rng.uniform(0, 4, 30000)
wall = np.stack([wx, wy, wz], axis=1)

# 车: 4x2x1.5 盒子的六个面均匀采样(表面!)
car_pts = []
for _ in range(10000):
    face = rng.integers(0, 6)
    u, v = rng.uniform(-1, 1, 2)
    if face == 0:   p = [5.0 + u * 2, 3.0 + v * 1, 1.5]
    elif face == 1: p = [5.0 + u * 2, 3.0 + v * 1, 0.0]
    elif face == 2: p = [5.0 + u * 2, 4.0, v * 0.75 + 0.75]
    elif face == 3: p = [5.0 + u * 2, 2.0, v * 0.75 + 0.75]
    elif face == 4: p = [7.0, 3.0 + u * 1, v * 0.75 + 0.75]
    else:           p = [3.0, 3.0 + u * 1, v * 0.75 + 0.75]
    car_pts.append(p)
car = np.array(car_pts) + rng.normal(0, 0.002, (10000, 3))

coord_surface = np.concatenate([ground, wall, car], axis=0)
ns = len(coord_surface)
print(f"\n原始: {ns} 点\n")

for gs in [0.05, 0.06, 0.12, 0.24, 0.48]:
    grid = np.floor(coord_surface / gs).astype(np.int64)
    key = grid[:, 0] * 10**10 + grid[:, 1] * 10**5 + grid[:, 2]
    nv = len(np.unique(key))
    print(f"  grid_size={gs:<5}: {nv:>7} 体素 (压缩 {ns/nv:.1f}x)")

print("\n  => 同样 12 万点,贴在表面上就能压 1.1x~5.4x。")
print("     想想为什么: 2D 表面嵌在 3D 格子里,格子是 3D 的," )
print("     同一片表面被相邻帧/相邻线反复扫到,点挤在同一格。")

# ============================================================
# 实验三：亲手实现"训练模式"GridSample(精简复刻) + 可视化
# ============================================================
print("\n" + "=" * 62)
print("实验三: 复刻 transform.py GridSample 训练模式")
print("=" * 62)

gs = 0.05
grid3 = np.floor(coord_surface / gs).astype(np.int64)
min3 = grid3.min(axis=0)
grid3_shifted = grid3 - min3              # 仓库: grid_coord -= min_coord
key3 = grid3_shifted[:, 0] * 10**10 + grid3_shifted[:, 1] * 10**5 + grid3_shifted[:, 2]

# --- 以下 6 行就是 transform.py:877-887 的核心 ---
idx_sort = np.argsort(key3)
key_sort = key3[idx_sort]
_, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
idx_select = (np.cumsum(np.insert(count, 0, 0)[0:-1])
              + np.random.randint(0, count.max(), count.size) % count)
idx_unique = idx_sort[idx_select]
# ----------------------------------------------------

sampled = coord_surface[idx_unique]
grid_coord_out = grid3_shifted[idx_unique]
print(f"\nGridSample 前: {coord_surface.shape[0]:>6} 点")
print(f"GridSample 后: {sampled.shape[0]:>6} 点 (grid_size={gs})")
print(f"grid_coord(整数格坐标,模型当'token 位置'用): {grid_coord_out.shape}")
print(f"每个体素随机选 1 个代表点 —— 训练模式引入数据扰动(相当于数据增强)")

# ---- 可视化对比: 左原(蓝) 右采样(橙),平移分开 ----
pcd_before = o3d.geometry.PointCloud()
pcd_before.points = o3d.utility.Vector3dVector(coord_surface)
pcd_before.paint_uniform_color([0.55, 0.7, 0.95])

pcd_after = o3d.geometry.PointCloud()
pcd_after.points = o3d.utility.Vector3dVector(sampled + np.array([55.0, 0, 0]))
pcd_after.paint_uniform_color([1.0, 0.6, 0.2])

print("\n弹出对比窗口: 左蓝=原始(12万), 右橙=GridSample 后")
print("观察: 右边变稀,但地面/墙/车的'轮廓结构'完好 —— 这就是意义:")
print("      用 1/1~1/5 的点保住几何,换来可控的计算量")
o3d.visualization.draw_geometries([pcd_before, pcd_after])

print("\n" + "=" * 62)
print("理解检查(答案写进 ONBOARDING_TODO.md):")
print("Q1: 为什么点云网络一定要先做 GridSample?")
print("    (提示: 12 万点 x 邻域 attention 的计算量 / 点数可控性)")
print("Q2: grid_size 变大 2 倍,体素数怎么变?为什么是'近似'关系?")
print("    (提示: 表面是 2D,格子是 3D,体素数 ~ 表面积/格子面积)")
print("=" * 62)
