import numpy as np
import matplotlib.pyplot as plt

# 定义数据
models = ['Scenario9+ResNet', 'Scenario32+ResNet', 'Scenario9+LeNet', 'Scenario32+LeNet']
top_k_acc = np.array([[47.17, 68.25, 81.00, 88.67, 94.25],
                     [28.88, 45.31, 52.23, 58.06, 62.06],
                     [47.83, 68.42, 81.00, 88.83, 93.42],
                     [12.90, 20.89, 26.57, 30.57, 34.72]])

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(5)
width = 0.2

for i, model in enumerate(models):
    ax.bar(x + i*width, top_k_acc[i], width=width, label=model)

ax.set_xlabel('Top-k')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Top-k Accuracy Comparison')
ax.set_xticks(x + width * (len(models)-1) / 2)
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax.legend()

plt.show()