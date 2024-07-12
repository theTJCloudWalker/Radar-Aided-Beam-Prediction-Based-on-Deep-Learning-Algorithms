import numpy as np
import matplotlib.pyplot as plt

models = ['RA64', 'RD', 'RC4']
top_k = [[0.3322091, 0.48735243, 0.56155145, 0.59865093, 0.6256324],
         [0.14839798, 0.18887016, 0.24283306, 0.28330523, 0.31365937],
         [0.17537943, 0.22259697, 0.2647555, 0.29679596, 0.33726814]]

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(5)
width = 0.25

for i, model in enumerate(models):
    ax.bar(x + i*width, top_k[i], width=width, label=model)

ax.set_xlabel('Top-k')
ax.set_ylabel('Accuracy')
ax.set_title('Top-k Accuracy Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax.legend()

plt.show()