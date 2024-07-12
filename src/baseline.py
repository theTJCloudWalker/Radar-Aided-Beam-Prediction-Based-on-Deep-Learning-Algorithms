import os
import numpy as np
import torch

from src.data import data_loader, preprocessing_torch, preprocessing_cupy

project_root_dir = '/mnt/c/Users/22006/OneDrive - 80shgy/last dance/ML-based-beam-prediction'

# 定义训练集和测试集的比例
train_ratio = 0.7
test_ratio = 1 - train_ratio

scenario_id = 9
print("Starting load data : scenario", scenario_id)
X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)
print(X.shape)


# win_type = 'kaiser'
# print(win_type)
# X = preprocessing_torch.window(X, type=win_type)


# 0: range_angle 1: range_doppler 3: radar_cube
pp_type = 0

processing_types = ["range_angle", "range_doppler", "radar_cube"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Starting prepropcessing : scenario", scenario_id)
# if processing_types[pp_type] == "range_angle":
#     X = preprocessing_torch.range_angle_map(X, fft_size=128)
if processing_types[pp_type] == "range_angle":
    X = preprocessing_cupy.range_angle_map(X, fft_size=64)
elif processing_types[pp_type] == "range_doppler":
    X = preprocessing_cupy.range_doppler_map(X, fft_size=X.shape[3])

num_samples = X.shape[0]
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

# 计算切分数据集的索引
num_train = int(train_ratio * num_samples)
num_test = int(test_ratio * num_samples)

# 按比例切分数据集
x_train, x_test = np.split(X, [num_train])
y_train, y_test = np.split(Y, [num_train])

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

num_training_data = x_train.shape[0]
num_test_data = x_test.shape[0]

# %%
number_of_beams = 64

v_train = torch.zeros((num_training_data, 1), dtype=torch.long, device=device)

for i in range(num_training_data):
    v_train[i] = x_train[i, 0, :, :].argmax()

# Look-up Table Generation
image = -1 * torch.ones((np.prod(x_test.shape[1:]), number_of_beams+1), dtype=torch.long, device=device)

for i in range(num_training_data):
    image[v_train[i].long(), y_train[i]] += 1

my_map_values, my_map = image.max(axis=1)
my_map[(my_map_values == -1)] = -1  # If there is no available data don't make a prediction

pred_train = my_map[v_train]
accuracy_train = (y_train.reshape(-1, 1) == pred_train).float().mean().cpu().item()
print(processing_types[pp_type])
print('Training Accuracy: %.2f%%' % (accuracy_train * 100))

v_test = torch.zeros((num_test_data, 5), dtype=torch.long, device=device)
pred_test = torch.zeros((num_test_data, 1), dtype=torch.long, device=device)

for i in range(num_test_data):
    v_test[i, 0] = x_test[i, 0, :, :].argmax()
    pred_test[i] = my_map[v_test[i, 0]]

    j = 1
    jc = 0
    cur_data = x_test[i, 0, :, :].flatten().sort(descending=True)[1]
    while j < 5 and jc < x_test.shape[0]:
        if not torch.sum(my_map[v_test[i, :]] == my_map[cur_data[jc]]):
            v_test[i, j] = cur_data[jc]
            j += 1
        jc += 1

pred_test = my_map[v_test]
accuracy_test = (y_test.reshape(-1, 1) == pred_test).float().mean(dim=0).cpu()
top_k_acc = accuracy_test.cumsum(0)

print('Top-k Test Accuracy: ', end='')
print(top_k_acc.numpy())

results = {}
results['Method'] = 'lookup_table'
results['Training Accuracy'] = accuracy_train
results['Top-k Accuracy'] = str(top_k_acc.numpy())
results['Number of Parameters'] = np.prod(my_map.shape)
print(results)
