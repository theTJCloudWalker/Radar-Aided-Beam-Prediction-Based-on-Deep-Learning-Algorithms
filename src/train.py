import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.data import data_loader, preprocessing_torch, preprocessing_cupy
from src.utils import network_func
from src.models import LeNet, AlexNet, ResNet

project_root_dir = '/mnt/c/Users/22006/OneDrive - 80shgy/last dance/ML-based-beam-prediction'
#torch.backends.cudnn.benchmark = True

# 定义训练集和测试集的比例
train_ratio = 0.7
test_ratio = 1 - train_ratio

# 超参数设置
num_epochs = 40
learning_rate = 0.001
# total=int(np.ceil(X_train.shape[0] / batch_size)
batch_size = 32
#batch_size = 32
#total=int(np.ceil(X_train.shape[0] / batch_size)
scenario_id = 9
print("Starting load data : scenario", scenario_id)
X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)
print(X.shape)

# 0: range_angle 1: range_doppler 2: radar_cube
pp_type = 0

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

# 将数据集按7:2:1的比例拆分
x_train, x_val_test, y_train, y_val_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.67, random_state=42)
print(x_train.shape)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

num_training_data = x_train.shape[0]
num_test_data = x_test.shape[0]
num_val_data = x_val.shape[0]

# print(num_training_data, num_test_data, num_val_data)

#
number_of_beams = 64

# Training

rng_seed = 0

# Model Save Folder
folder_name = 'type%i_batchsize%i_seed%i_epoch%i_scenario%i' % (pp_type, batch_size, rng_seed, num_epochs, scenario_id)
models_directory = os.path.abspath('../saved_models/')
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
c = 0
while os.path.exists(os.path.join(models_directory, folder_name + '_v%i' % c, '')):
    c += 1
model_directory = os.path.join(models_directory, folder_name + '_v%i' % c, '')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
print('Saving the models to %s' % models_directory)

# Reproducibility
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)
torch.backends.cudnn.deterministic = True

neural_nets = []

# Neural Network
# net = neural_nets[pp_type]()
#net = LeNet.LeNet_RangeAngle()
# net = LeNet.LeNet_RangeAngle_512()

#net = AlexNet.AlexNet_RangeAngle_512()

net = ResNet.ResNet34().to(device)

# 将张量移动到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Training Settings
net.to(device)
criterion = torch.nn.CrossEntropyLoss()  # Training Criterion
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)  # Optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_loss = np.zeros((num_epochs))
train_acc = np.zeros((num_epochs))
val_loss = np.zeros((num_epochs))
val_acc = np.zeros((num_epochs))

early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0

# Epochs
for epoch in range(num_epochs):
    print('Epoch %i/%i:' % (epoch + 1, num_epochs), flush=True)

    train_loss[epoch], train_acc[epoch] = network_func.train_loop(x_train, y_train, net, optimizer, criterion, device,
                                                                  batch_size=batch_size)
    val_loss[epoch], val_acc[epoch] = network_func.eval_loop(x_val, y_val, net, criterion, device,
                                                             batch_size=batch_size)

    # Save the best model
    if val_loss[epoch] <= np.min(val_loss[:epoch] if epoch > 0 else val_loss[epoch]):
        # print('Saving model..')
        torch.save(net.state_dict(), os.path.join(model_directory, 'model_best.pth'))

    if val_loss[epoch] < best_val_loss:
        best_val_loss = val_loss[epoch]
        patience_counter = 0
        torch.save(net.state_dict(), os.path.join(model_directory, 'model_best.pth'))
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

    scheduler.step()


torch.save(net.state_dict(), os.path.join(model_directory, 'model_final.pth'))

print('Finished Training')

print('Testing..')
topk = 5

y, y_hat, network_time_per_sample = network_func.test_loop(x_test, y_test, net, device,
                                                           model_path=os.path.join(model_directory,
                                                                                   'model_best.pth'))  # Best model with minimum lost
topk_acc_best, beam_dist_best = network_func.evaluate_predictions(y, y_hat, k=topk)
print('Best model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_best * 100))
print('Beam distance: %.2f' % beam_dist_best)

y_final, y_hat_final, network_time_per_sample_final = network_func.test_loop(x_test, y_test, net, device,
                                                                             model_path=os.path.join(model_directory,
                                                                                                     'model_final.pth'))  # Last Epoch
topk_acc_final, beam_dist_final = network_func.evaluate_predictions(y_final, y_hat_final, k=topk)
print('Final model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_final * 100))
print('Beam distance: %.2f' % beam_dist_final)
