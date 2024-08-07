import matplotlib.pyplot as plt
import numpy as np

# 数据
best_model_top_k = [47.17, 68.25, 81.00, 88.67, 94.25]
best_model_kaiser1_top_k = [47.17, 69.42, 81.75, 89.67, 94.17]
best_model_kaiser2_top_k = [47.75, 68.92, 82.25, 89.58, 94.08]
best_model_kaiser5_top_k = [48.42, 69.58, 82.33, 90.00, 94.50]
best_model_kaiser10_top_k = [48.92, 68.58, 81.33, 89.00, 94.67]

final_model_top_k = [46.92, 68.00, 81.33, 88.50, 93.67]
final_model_kaiser1_top_k = [47.17, 69.42, 81.75, 89.67, 94.17]
final_model_kaiser2_top_k = [47.75, 68.92, 82.25, 89.58, 94.08]
final_model_kaiser5_top_k = [48.25, 69.83, 81.83, 89.92, 94.50]
final_model_kaiser10_top_k = [48.25, 68.42, 80.92, 89.25, 94.50]

# 创建图像
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Best model
ax[0].bar(np.arange(5) - 0.6, best_model_top_k, 0.2, label='None', color='#03A9F4')
ax[0].bar(np.arange(5) - 0.4, best_model_kaiser1_top_k, 0.2, label='Kaiser β=1', color='#8BC34A')
ax[0].bar(np.arange(5) - 0.2, best_model_kaiser2_top_k, 0.2, label='Kaiser β=2', color='#FFC107')
ax[0].bar(np.arange(5), best_model_kaiser5_top_k, 0.2, label='Kaiser β=5', color='#F44336')
ax[0].bar(np.arange(5) + 0.2, best_model_kaiser10_top_k, 0.2, label='Kaiser β=10', color='#673AB7')
ax[0].set_title('Best Model')

# Final model
ax[1].bar(np.arange(5) - 0.6, final_model_top_k, 0.2, label='None', color='#03A9F4')
ax[1].bar(np.arange(5) - 0.4, final_model_kaiser1_top_k, 0.2, label='Kaiser β=1', color='#8BC34A')
ax[1].bar(np.arange(5) - 0.2, final_model_kaiser2_top_k, 0.2, label='Kaiser β=2', color='#FFC107')
ax[1].bar(np.arange(5), final_model_kaiser5_top_k, 0.2, label='Kaiser β=5', color='#F44336')
ax[1].bar(np.arange(5) + 0.2, final_model_kaiser10_top_k, 0.2, label='Kaiser β=10', color='#673AB7')
ax[1].set_title('Final Model')

ax[0].set_xlabel('Top-k Accuracy')
ax[0].set_ylabel('Accuracy (%)')
ax[1].set_xlabel('Top-k Accuracy')
ax[1].set_ylabel('Accuracy (%)')

ax[0].set_xticks(np.arange(5))
ax[0].set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax[1].set_xticks(np.arange(5))
ax[1].set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])

ax[0].legend()
ax[1].legend()

plt.show()