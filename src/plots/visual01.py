import matplotlib.pyplot as plt
import numpy as np


# 数据
scenarios = ['Scenario9', 'Scenario32', 'Scenario33', 'Scenario35']
window_functions = ['None', 'Kaiser (beta=5)', 'Rectangular', 'Hamming', 'Hanning', 'Blackman']
accuracies = [
    [46.51, 46.53, 46.51, 46.51, 46.77, 46.42],
    [48.67, 49.20, 48.67, 49.34, 49.65, 50.09],
    [54.06, 55.20, 54.06, 55.28, 55.78, 56.07],
    [49.46, 49.98, 49.46, 50.02, 49.98, 50.59]
]

# 设置图表
fig, ax = plt.subplots(figsize=(12, 8))

# 计算每个条的位置
bar_width = 0.15
index = np.arange(len(window_functions))

# 绘制每个场景的柱状图
for i, scenario in enumerate(scenarios):
    bars = ax.bar(index + i * bar_width, accuracies[i], bar_width, label=scenario)
    # 添加数据标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', ha='center', va='bottom')

# 设置图表标题和标签
ax.set_xlabel('Window Function')
ax.set_ylabel('Training Accuracy (%)')
ax.set_title('Training Accuracy under Different Window Functions for Various Scenarios')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(window_functions)
ax.legend()

# 显示图表
plt.ylim([45, 60])  # 根据数据调整Y轴范围
plt.show()
