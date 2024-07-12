import matplotlib.pyplot as plt
import numpy as np

# 数据
window_sizes = ['RA4', 'RA64', 'RA128', 'RA256', 'RA512', 'RA1024', 'RA2048']
scenario9_accuracies = [31.36, 46.51, 50.20, 55.25, 62.37, 72.21, 80.76]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

index = np.arange(len(window_sizes))
bar_width = 0.45
ax.bar(index, scenario9_accuracies, width=bar_width)
# 在柱子上方显示数值
for i, acc in enumerate(scenario9_accuracies):
    ax.text(i, acc + 1, f"{acc:.2f}%", ha='center', va='bottom')

ax.set_xlabel('Super Resolution Size')
ax.set_ylabel('Training Accuracy (%)')
ax.set_title('Scenario9 Training Accuracy for SR Sizes')
ax.set_xticks(index)
ax.set_xticklabels(window_sizes)

plt.tight_layout()
plt.show()