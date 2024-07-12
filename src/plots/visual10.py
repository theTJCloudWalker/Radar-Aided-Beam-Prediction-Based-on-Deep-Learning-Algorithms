import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Best Model', 'Final Model']
scenarios = ['Scenario9', 'Scenario32', 'Scenario35']
best_model = [93.42, 34.72, 38.5]
final_model = [93.75, 34.72, 38.5]

# 设置柱状图的宽度
bar_width = 0.2
index = np.arange(len(labels))

# 偏移量
offset = np.array([-bar_width, 0, bar_width])

# 创建图表
fig, ax = plt.subplots()

# 绘制柱状图
for i in range(len(scenarios)):
    ax.bar(index + offset[i], [best_model[i], final_model[i]], bar_width, label=scenarios[i])

# 添加标题和标签
ax.set_xlabel('Model Type')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparison of Best Model and Final Model Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.legend()

# 在每个柱子上标注数值
for i in range(len(scenarios)):
    ax.annotate(f'{best_model[i]:.2f}%',
                xy=(index[0] + offset[i], best_model[i]),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    ax.annotate(f'{final_model[i]:.2f}%',
                xy=(index[1] + offset[i], final_model[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# 显示图表
plt.show()
