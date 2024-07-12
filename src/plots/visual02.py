import matplotlib.pyplot as plt
import numpy as np

# 数据
scenarios = ['Scenario9', 'Scenario32', 'Scenario33', 'Scenario35']
window_functions = ['None', 'Kaiser (beta=5)', 'Rectangular', 'Hamming', 'Hanning', 'Blackman']
top5_accuracies = [
    [64.52514, 64.58685, 62.56324, 65.09275, 65.93592, 65.09275],
    [31.204945, 27.909374, 31.204945, 27.497426, 28.01236, 27.703398],
    [29.288703, 26.94561, 29.288703, 26.359832, 27.196655, 26.192468],
    [29.288703, 26.94561, 29.288703, 26.359832, 27.196655, 26.192468]
]

# 设置图表
fig, ax = plt.subplots(figsize=(12, 8))

# 计算每个条的位置
bar_width = 0.15
index = np.arange(len(window_functions))

# 绘制每个场景的柱状图
for i, scenario in enumerate(scenarios):
    bars = ax.bar(index + i * bar_width, top5_accuracies[i], bar_width, label=scenario)
    # 添加数据标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', ha='center', va='bottom')

# 设置图表标题和标签
ax.set_xlabel('Window Function')
ax.set_ylabel('Top-5 Accuracy (%)')
ax.set_title('Top-5 Accuracy under Different Window Functions for Various Scenarios')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(window_functions)
ax.legend()

# 显示图表
plt.ylim([0, 70])  # 根据数据调整Y轴范围
plt.show()
