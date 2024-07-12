import matplotlib.pyplot as plt
import numpy as np

# 数据
window_types = ['None', 'kaiser beta=5', 'rectangular', 'hamming', 'hanning', 'Blackman']
best_model = [88.2, 91.4, 92.58, 91.06, 90.89, 90.89]
final_model = [94.27, 94.44, 94.27, 93.76, 93.59, 94.1]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

index = np.arange(len(window_types))
bar_width = 0.35

best_bars = ax.bar(index - bar_width/2, best_model, width=bar_width, label='Best Model')
final_bars = ax.bar(index + bar_width/2, final_model, width=bar_width, label='Final Model')

# 添加准确率数值
for i, bar in enumerate(best_bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{best_model[i]:.2f}%", ha='center', va='bottom')

for i, bar in enumerate(final_bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{final_model[i]:.2f}%", ha='center', va='bottom')

ax.set_xlabel('Window Type')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy for Different Window Types With LeNet-RA64')
ax.set_xticks(index)
ax.set_xticklabels(window_types, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()