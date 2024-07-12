import numpy as np
import matplotlib.pyplot as plt

models = ['RA64', 'RD', 'RC4']
train_acc = [46.51, 16.19, 19.79]
num_params = [16384, 131072, 32768]

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(models, train_acc, width=0.2)
ax.set_xlabel('Model')
ax.set_ylabel('Training Accuracy (%)')
ax.set_title('Training Accuracy vs Number of Parameters')

ax2 = ax.twinx()
ax2.plot(models, num_params, 'ro-')
ax2.set_ylabel('Number of Parameters')
ax2.set_yscale('log')

plt.show()