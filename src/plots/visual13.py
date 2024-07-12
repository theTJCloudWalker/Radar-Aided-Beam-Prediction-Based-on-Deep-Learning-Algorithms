# import matplotlib.pyplot as plt
#
# # 训练和验证数据
# epochs = list(range(1, 41))
# train_accuracy = [27.4, 34.8, 36.4, 38.5, 39.9, 41.8, 41.4, 42.7, 42.7, 46.3, 51.2, 54.1, 55.1, 56.7, 57.3, 58.1, 58.6, 59.5, 62.1, 62.2, 65.2, 66.6, 67.2, 67.8, 67.5, 68.1, 68.4, 68.1, 68.5, 68.7, 69.1, 69.5, 69.4, 68.7, 69.3, 70.6, 69.3, 69.4, 69.3, 69.3]
# train_loss = [2.69, 2.03, 1.93, 1.82, 1.79, 1.72, 1.68, 1.64, 1.62, 1.53, 1.34, 1.26, 1.23, 1.17, 1.16, 1.14, 1.11, 1.07, 1.04, 1.01, 0.953, 0.925, 0.915, 0.915, 0.909, 0.905, 0.893, 0.905, 0.884, 0.878, 0.861, 0.867, 0.875, 0.882, 0.866, 0.853, 0.863, 0.865, 0.872, 0.874]
# val_accuracy = [27.46, 6.95, 14.24, 20.68, 36.10, 39.83, 36.95, 14.75, 40.00, 20.68, 34.92, 38.47, 44.92, 45.08, 44.92, 42.88, 29.15, 42.03, 33.90, 42.71, 40.17, 40.17, 43.56, 40.51, 41.53, 38.81, 41.69, 42.20, 41.36, 42.37, 40.68, 41.69, 41.19, 40.85, 41.69, 41.19, 41.02, 40.34, 41.69, 41.69]
# val_loss = [2.3799, 3.5104, 3.3619, 2.4218, 2.0182, 1.7800, 1.8622, 5.0906, 1.7332, 3.0538, 2.0014, 1.8629, 1.5354, 1.5502, 1.6708, 1.8352, 2.3337, 1.6935, 2.1922, 1.7464, 1.7793, 1.7926, 1.7604, 1.7960, 1.7832, 1.8347, 1.8009, 1.8133, 1.8110, 1.8241, 1.8242, 1.8198, 1.8281, 1.8201, 1.8338, 1.8247, 1.8297, 1.8332, 1.8322, 1.8317]
#
# # 绘制训练和验证准确度曲线
# plt.figure(figsize=(14, 6))
#
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_accuracy, label='Training Accuracy')
# plt.plot(epochs, val_accuracy, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
#
# # 绘制训练和验证损失曲线
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
########################################################################################################
# import matplotlib.pyplot as plt
#
# # 假设这些数据是从你的训练日志中提取的
# train_accuracies = [25.9, 33.8, 36.2, 38.3, 39.5, 41.5, 40, 43.2, 43.9, 45.2, 51.7, 54, 55, 57.6, 58.5, 60.1, 60.9, 61.8, 65.3, 65.9, 70.6, 72, 73.2, 73.9, 74, 74.1, 74.8, 73.2, 75.3, 76.1, 76.6, 77.2, 76.7, 77.1, 77.2, 77.7, 76.3, 77.2, 76.8, 76.7]
# val_accuracies = [3.90, 16.10, 36.78, 35.76, 41.19, 43.90, 39.15, 43.73, 23.39, 42.37, 44.75, 47.80, 46.61, 44.58, 36.78, 42.20, 41.86, 35.42, 41.53, 40.85, 41.86, 40.00, 43.05, 40.51, 39.66, 41.02, 41.53, 39.83, 40.00, 41.02, 40.85, 41.36, 42.88, 41.36, 41.02, 41.53, 40.85, 40.51, 40.00, 39.32]
# train_losses = [2.74, 2.07, 1.92, 1.85, 1.77, 1.71, 1.69, 1.63, 1.57, 1.53, 1.33, 1.25, 1.21, 1.16, 1.12, 1.08, 1.05, 1.02, 0.953, 0.939, 0.841, 0.813, 0.783, 0.773, 0.767, 0.765, 0.744, 0.756, 0.725, 0.714, 0.703, 0.691, 0.705, 0.705, 0.697, 0.687, 0.702, 0.697, 0.707, 0.7]
# val_losses = [5.8705, 3.1682, 1.8303, 1.8714, 1.7804, 1.6039, 1.9445, 1.6028, 3.7932, 1.5882, 1.5244, 1.4953, 1.5577, 1.5860, 1.9369, 1.7142, 1.6857, 2.1638, 1.7744, 1.8141, 1.8062, 1.8393, 1.8132, 1.8693, 1.8670, 1.8730, 1.8910, 1.9333, 1.9249, 1.9184, 1.9334, 1.9191, 1.9063, 1.9337, 1.9435, 1.9412, 1.9336, 1.9499, 1.9470, 1.9820]
#
# epochs = range(1, 41)
#
# # 绘制训练和验证精确度曲线
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_accuracies, label='Training Accuracy')
# plt.plot(epochs, val_accuracies, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # 绘制训练和验证损失曲线
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_losses, label='Training Loss')
# plt.plot(epochs, val_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
#
# # Assuming you have the training and validation accuracy and loss values stored in the following variables
# train_acc = [25.9, 33.8, 36.2, 38.3, 39.5, 41.5, 40.0, 43.2, 43.9, 45.2, 51.7, 54.0, 55.0, 57.6, 58.5, 60.1, 60.9]
# train_loss = [2.74, 2.07, 1.92, 1.85, 1.77, 1.71, 1.69, 1.63, 1.57, 1.53, 1.33, 1.25, 1.21, 1.16, 1.12, 1.08, 1.05]
# val_acc = [3.90, 16.10, 36.78, 35.76, 41.19, 43.90, 39.15, 43.73, 23.39, 42.37, 44.75, 47.80, 46.61, 44.58, 36.78, 42.20, 41.86]
# val_loss = [5.8705, 3.1682, 1.8303, 1.8714, 1.7804, 1.6039, 1.9445, 1.6028, 3.7932, 1.5882, 1.5244, 1.4953, 1.5577, 1.5860, 1.9369, 1.7142, 1.6857]
#
# # Create a figure and two subplots for accuracy and loss
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
#
# # Plot the training and validation accuracy
# ax1.plot(range(1, len(train_acc)+1), train_acc, label='Training')
# ax1.plot(range(1, len(val_acc)+1), val_acc, label='Validation')
# ax1.set_title('Accuracy')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Accuracy (%)')
# ax1.legend()
#
# # Plot the training and validation loss
# ax2.plot(range(1, len(train_loss)+1), train_loss, label='Training')
# ax2.plot(range(1, len(val_loss)+1), val_loss, label='Validation')
# ax2.set_title('Loss')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Loss')
# ax2.legend()
#
# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.5)
#
# # Show the plot
# plt.show()
import matplotlib.pyplot as plt

# 假设这些数据是从你的训练日志中提取的
train_accuracies = [11.1, 12.3, 12.9, 12.1, 12.7, 14.1, 14.1]
val_accuracies = [14.69, 14.69, 14.69, 4.69, 5.62, 2.81, 0.62]
train_losses = [3.74, 3.65, 3.6, 3.55, 3.51, 3.41, 3.32]
val_losses = [3.8133, 3.7029, 5.5661, 4.1007, 5.2446, 12.6144, 85.3181]

epochs = range(1, len(train_accuracies) + 1)

# 绘制训练和验证精确度曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制训练和验证损失曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
