import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.data import data_loader, preprocessing_torch, preprocessing_cupy
#matplotlib.use('TkAgg')
project_root_dir = "C:\\Users\\cloudwalker\\OneDrive - 80shgy\\last dance\\ML-based-beam-prediction"

# 定义训练集和测试集的比例
train_ratio = 0.7
test_ratio = 1 - train_ratio

scenario_id = 33
print("Starting load data : scenario", scenario_id)
X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)

print(Y.shape)

# 创建一个长度为64的字典，其中键是0到63的整数，值为0
map_64 = {i: 0 for i in range(65)}

for i in range(len(Y)):
    map_64[Y[i]] += 1

# 提取键和值
keys = list(map_64.keys())
values = list(map_64.values())

# 绘制柱状图
plt.figure(figsize=(15, 7))
plt.bar(keys, values)
plt.xlabel('Keys')
plt.ylabel('Values')
plt.title('Bar Chart of map_64 of scenario {}'.format(scenario_id))
plt.xticks(keys, rotation='vertical')  # 显示所有的键，并将它们垂直显示
plt.grid(axis='y')

# 显示图表
plt.show()