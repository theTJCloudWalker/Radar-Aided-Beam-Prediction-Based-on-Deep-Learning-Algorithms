import numpy as np
import matplotlib.pyplot as plt

# Define the data
models = ['RA512', 'RD512', 'RC4']
best_model_acc = np.array([[47.17, 68.25, 81.00, 88.67, 94.25],
                          [42.42, 64.83, 76.83, 85.92, 91.17],
                          [47.08, 68.83, 82.08, 88.42, 93.83]]).T
final_model_acc = np.array([[46.92, 68.00, 81.33, 88.50, 93.67],
                           [42.67, 63.92, 77.58, 85.83, 91.50],
                           [41.83, 63.67, 76.00, 83.83, 90.33]]).T

# Best Model Plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(5)  # Number of top-k values
width = 0.25

ax1.bar(x - width, best_model_acc[:, 0], width=width, label='RA512')
ax1.bar(x, best_model_acc[:, 1], width=width, label='RD512')
ax1.bar(x + width, best_model_acc[:, 2], width=width, label='RC4')

ax1.set_xlabel('Top-k')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Best Model - Top-k Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax1.legend()

# Final Model Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(x - width, final_model_acc[:, 0], width=width, label='RA512')
ax2.bar(x, final_model_acc[:, 1], width=width, label='RD512')
ax2.bar(x + width, final_model_acc[:, 2], width=width, label='RC4')

ax2.set_xlabel('Top-k')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Final Model - Top-k Accuracy Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax2.legend()

plt.show()