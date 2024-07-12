import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['Range-Angle', 'Range-Velocity', 'Radar-Cube']
top_k = [1, 2, 3, 4, 5]
accuracies = [
    [46.21, 66.44, 80.44, 90.05, 93.76],
    [40.47, 59.70, 73.52, 82.80, 88.87],
    [40.30, 61.21, 74.20, 85.16, 90.22]
]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

index = np.arange(len(top_k))
bar_width = 0.2
x_offset = -0.3

for i, model in enumerate(models):
    ax.bar(index + x_offset + i*bar_width, accuracies[i], width=bar_width, label=model)

ax.set_xlabel('ALEXNET On Scenario 9')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy for Different preprocess types and Top-K')
ax.set_xticks(index + (len(models)-1)*bar_width/2)
ax.set_xticklabels([f"Top-{k}" for k in top_k])
ax.legend()

plt.tight_layout()
plt.show()