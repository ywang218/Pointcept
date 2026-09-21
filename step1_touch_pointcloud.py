"""
Step 1: 用 open3d 亲手"摸"一次点云
模拟一帧 64 线 LiDAR 扫描（约 12 万点），弹出可旋转的 3D 窗口观察。

运行: python -X utf8 step1_touch_pointcloud.py
操作: 鼠标左键拖动旋转 / 滚轮缩放 / 中键平移
"""
import numpy as np
import open3d as o3d

rng = np.random.default_rng(42)
n = 120000  # 一帧 SemanticKITTI 大约 12 万点

# ---- 模拟 LiDAR 扫描几何 ----
# LiDAR 是极坐标传感器：绕 z 轴 360° 旋转打线，每条线一个仰角
# 所以点分布天然"近密远疏"（角度分辨率固定，距离越远弧长越大点越稀）
angles = rng.uniform(0, 2 * np.pi, n)        # 水平角 [0, 2π)
r = np.sqrt(rng.uniform(0, 1, n)) * 35.0     # 距离 0~35m，sqrt 使近处更密
z = rng.uniform(-2.0, 3.0, n)                # 高度（模拟多条扫描线的高度展开）

x = r * np.cos(angles)
y = r * np.sin(angles)
coord = np.stack([x, y, z], axis=1)          # [N, 3] —— 点云的全部"几何"

print(f"点数: {coord.shape[0]}, 坐标范围: "
      f"x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}] z[{z.min():.1f},{z.max():.1f}]")

# ---- 用高度给点上色（真实 LiDAR 可视化的常用方式）----
# 语义分割的任务就是给每个点上"类别颜色"——现在是按高度假着色，
# 训练好的模型输出就是按"语义类别"着色
z_norm = (z - z.min()) / (z.max() - z.min())                    # 0~1
colors = np.stack([z_norm, 1 - z_norm * 0.5, 0.3 + z_norm * 0.4], axis=1)  # 低处蓝->高处黄

# ---- 构造 open3d 点云对象并显示 ----
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("弹出 3D 窗口... 左键旋转 / 滚轮缩放 / 中键平移，关闭窗口后脚本结束")
o3d.visualization.draw_geometries([pcd])

# ---- 对比实验：再来一个"高斯团"（模拟单个物体，如一辆车 2~3 千点）----
print("\n再对比看一个'物体级'点云（模拟一辆车的点数密度）...")
m = 3000
car = rng.normal(0, [1.0, 1.0, 0.4], (m, 3))    # 4.5m x 2m x 1.5m 的椭球状点团
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(car)
pcd2.paint_uniform_color([0.9, 0.3, 0.2])
o3d.visualization.draw_geometries([pcd2])
print("Step 1 完成。")
