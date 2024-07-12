import matplotlib.pyplot as plt
import numpy as np

# 数据
window_sizes = ['RA4', 'RA64', 'RA128', 'RA256', 'RA512', 'RA1024', 'RA2048']
scenario9_topk_accuracies = [
    [
        [0.24581005, 0.36089385, 0.4162011, 0.44581005, 0.46871507],
        [0.3322091, 0.48735243, 0.56155145, 0.59865093, 0.6256324],
        [0.33895448, 0.5143339, 0.61214167, 0.68296796, 0.72681284],
        [0.306914, 0.5059022, 0.608769, 0.7133221, 0.762226],
        [0.25632378, 0.48229343, 0.6239461, 0.7234402, 0.79426646],
        [0.19730186, 0.42833054, 0.60370994, 0.7015177, 0.767285],
        [0.09553073, 0.35810053, 0.51284915, 0.64134073, 0.724581]
     ]
]

# Convert to percentage strings
def convert_to_percentage(accuracies):
    return [[f"{acc * 100:.2f}%" for acc in top] for top in accuracies[0]]

top5_accuracies_percent = convert_to_percentage(scenario9_topk_accuracies)

# Plot the top-k accuracy for each window function
fig, ax = plt.subplots(figsize=(12, 8))

index = np.arange(5)
bar_width = 0.12

for i, wf in enumerate(window_sizes):
    ax.bar(index + i * bar_width, [float(acc[:-1]) for acc in top5_accuracies_percent[i]], bar_width, label=wf)

ax.set_xlabel('Scenario9 Top-k Accuracy of Baseline ')
ax.set_ylabel('top-k Accuracy (%)')
ax.set_title('Top-5 Accuracy for Different SR sizes')
ax.set_xticks(index + (len(window_sizes) - 1) * bar_width / 2)
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax.legend()

plt.tight_layout()
plt.show()