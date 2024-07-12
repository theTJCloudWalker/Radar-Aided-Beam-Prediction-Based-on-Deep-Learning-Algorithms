import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.data import data_loader, preprocessing_torch, preprocessing_cupy
from src.utils import network_func
from src.models import LeNet, AlexNet, ResNet

project_root_dir = "C:\\Users\\cloudwalker\\OneDrive - 80shgy\\last dance\\ML-based-beam-prediction"
#torch.backends.cudnn.benchmark = True



#total=int(np.ceil(X_train.shape[0] / batch_size)
scenario_id = 32
print("Starting load data : scenario", scenario_id)
X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)
print(X.shape)

# 0: range_angle 1: range_doppler 2: radar_cube
pp_type = 2

processing_types = ["range_angle", "range_doppler", "radar_cube"]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

print("Starting prepropcessing : scenario", scenario_id)
# if processing_types[pp_type] == "range_angle":
#     X = preprocessing_torch.range_angle_map(X, fft_size=128)
if processing_types[pp_type] == "range_angle":
    X = preprocessing_cupy.range_angle_map(X, fft_size=512)
elif processing_types[pp_type] == "range_doppler":
    X = preprocessing_cupy.range_doppler_map(X, fft_size=X.shape[3])
elif processing_types[pp_type] == "radar_cube":
    X = preprocessing_cupy.radar_cube_map(X, fft_size=4)



num_samples = X.shape[0]
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

# 模型目录和路径设置
model_directory = os.path.join(project_root_dir, '\\src\\saved_models', 'type2_batchsize32_seed0_epoch40_v0')  # 修改为实际路径
best_model_path = os.path.join(model_directory, 'model_best.pth')

# 初始化模型并加载参数
net = ResNet.ResNet34().to(device)
net.load_state_dict(torch.load(best_model_path))
net.eval()  # 设置模型为评估模式

# 测试模型
print('Testing best model..')
topk = 5

y, y_hat, network_time_per_sample = network_func.test_loop(X, Y, net, device)
topk_acc_best, beam_dist_best = network_func.evaluate_predictions(y, y_hat, k=topk)
print('Best model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' % acc for acc in (topk_acc_best * 100)]))
print('Beam distance: %.2f' % beam_dist_best)