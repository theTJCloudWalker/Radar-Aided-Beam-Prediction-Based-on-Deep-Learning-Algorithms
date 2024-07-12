import matplotlib.pyplot as plt
import numpy as np

# 数据
range_angle = [42.50, 61.05, 72.68, 82.12, 88.20]
range_velocity = [41.18, 59.87, 71.67, 82.12, 88.20]
radar_cube = [41.82, 63.24, 75.04, 86.17, 91.57]

x = np.arange(5)  # x轴刻度
width = 0.25  # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
ax.bar(x - width, range_angle, width, label='Range-Angle-64')
ax.bar(x, range_velocity, width, label='Range-Velocity')
ax.bar(x + width, radar_cube, width, label='Radar-Cube-4')

# 设置x轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
plt.xticks(rotation=90)

# 添加标题和坐标轴标签
ax.set_title('Best Model Top-k Accuracy')
ax.set_ylabel('Accuracy (%)')
ax.legend()

plt.show()

# 数据
range_angle = [45.87, 68.13, 79.43, 89.21, 94.27]
range_velocity = [40.98, 59.53, 73.19, 83.47, 88.70]
radar_cube = [42.50, 62.06, 76.56, 87.18, 92.07]

x = np.arange(5)  # x轴刻度
width = 0.25  # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
ax.bar(x - width, range_angle, width, label='Range-Angle-64')
ax.bar(x, range_velocity, width, label='Range-Velocity')
ax.bar(x + width, radar_cube, width, label='Radar-Cube-4')

# 设置x轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
plt.xticks(rotation=90)

# 添加标题和坐标轴标签
ax.set_title('Final Model Top-k Accuracy')
ax.set_ylabel('Accuracy (%)')
ax.legend()

plt.show()