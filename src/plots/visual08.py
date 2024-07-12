import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置seaborn风格
sns.set(style="whitegrid")

# 数据
networks = ['Baseline', 'LeNet', 'AlexNet', 'GoogleNet', 'ResNet']
training_acc = [46.51, 94.27, 93.59, 93.59, 94.67]
test_acc = [62.56, 88.2, 93.76, 92.41, 94.92]

# 设置柱状图的宽度
bar_width = 0.15

# x轴的位置
x = np.arange(2)  # 对应 'training_acc' 和 'test_acc'

# 定义清新的颜色
colors = sns.color_palette("pastel", len(networks))

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.1

# 绘制每个网络的柱状图
for i, (network, color) in enumerate(zip(networks, colors)):
    bars = ax.bar(x + i * bar_width, [training_acc[i], test_acc[i]], bar_width, label=network, color=color)

    # 在每个柱状图上添加数据标签
    for bar, accuracy in zip(bars, [training_acc[i], test_acc[i]]):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# 添加标签、标题和图例
ax.set_ylabel('Top-5 Accuracy (%)')
ax.set_xticks(x + bar_width * (len(networks) - 1) / 2)
ax.set_xticklabels(['Training Top-5 Accuracy', 'Test Top-5 Accuracy'])
ax.set_title('Training and Test Top-5 Accuracy for Different Networks')
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
