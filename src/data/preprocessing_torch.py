import torch
import numpy as np
from tqdm import tqdm


def range_angle_map(data, fft_size=64):
    shape = data.shape
    # shape:[length, RX, samples, chirps]
    size = list(shape)
    size[1] = 1
    size[3] = size[2]
    size[2] = fft_size

    #[length, 1, 64, 256]
    result = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(shape[0])):
        result[i, 0, :, :] = range_angle_map_2d(data[i], fft_size)
    result = result.transpose([0, 1, 3, 2])
    return result


def range_angle_map_2d(data, fft_size=64):
    # shape:[RX, samples, chirps]
    data = torch.from_numpy(data)
    data = torch.fft.fft(data, axis=1)  # Range FFT
    # 突出特征
    data -= torch.mean(data, axis=2, keepdims=True)
    data = torch.fft.fft(data, n=fft_size, axis=0)  # Angle FFT
    data = torch.abs(data).sum(axis=2)  # Sum over velocity
    return data


def window(data, type):
    # Define the window functions
    # length : samples
    window_functions = {
        'Rectangular': np.ones(data.shape[2]),
        'Hamming': np.hamming(data.shape[2]),
        'Blackman': np.blackman(data.shape[2]),
        'Hanning': np.hanning(data.shape[2]),
        'kaiser': np.kaiser(data.shape[2], 5)
    }

    # 创建一个与雷达数据形状相同的数组来存储加窗后的数据，使用复数的实部
    data_windowed = np.zeros_like(data, dtype=np.complex64)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[3]):
                data_windowed[i, j, :, k] = data[i, j, :, k] * window_functions[type]
    return data_windowed


