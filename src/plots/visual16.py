import numpy as np
import matplotlib.pyplot as plt

# 假设你提供了以下数据
epochs = np.arange(1, 11)
train_acc = [46.51, 48.72, 50.91, 52.22, 53.45, 54.63, 55.31, 56.02, 56.74, 57.28]
train_loss = [1.23, 1.15, 1.08, 1.02, 0.96, 0.91, 0.87, 0.83, 0.79, 0.76]
val_acc = [44.82, 46.13, 47.91, 49.22, 50.56, 51.71, 52.41, 53.09, 53.75, 54.23]
val_loss = [1.31, 1.23, 1.16, 1.09, 1.03, 0.97, 0.93, 0.89, 0.85, 0.82]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制准确度曲线
ax1.plot(epochs, train_acc, label='Train Accuracy')
ax1.plot(epochs, val_acc, label='Val Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy Curves')
ax1.legend()

# 绘制损失曲线
ax2.plot(epochs, train_loss, label='Train Loss')
ax2.plot(epochs, val_loss, label='Val Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Curves')
ax2.legend()

plt.tight_layout()
plt.show()