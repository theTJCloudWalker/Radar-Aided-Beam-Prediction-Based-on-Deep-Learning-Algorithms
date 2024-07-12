import torch
import numpy as np
import cupy as cp
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


def range_doppler_map(data, fft_size=64):
    shape = data.shape
    size = list(shape)
    size[1] = 1
    size[3]=fft_size
    result = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        result[i, 0, :, :] = range_doppler_map_2d(data[i], fft_size)
    return result


def range_doppler_map_2d(data, fft_size):
    data = cp.asarray(data)
    data = cp.fft.fft(data, axis=1)  # Range FFT
    data -= cp.mean(data, 2, keepdims=True)
    data = cp.fft.fft(data, n=fft_size, axis=2) # Velocity FFT
    # data = np.fft.fftshift(data, axes=2)
    data = cp.abs(data).sum(axis=0)  # Sum over antennas
    data = cp.asnumpy(data)
    return data


def range_angle_map_2d(data, fft_size=64):
    # shape:[RX, samples, chirps]
    data = cp.asarray(data)
    data = cp.fft.fft(data, axis=1)  # Range FFT
    # 突出特征
    data -= cp.mean(data, axis=2, keepdims=True)
    data = cp.fft.fft(data, n=fft_size, axis=0)  # Angle FFT
    data = cp.abs(data).sum(axis=2)  # Sum over velocity
    data = cp.asnumpy(data)
    return data


def radar_cube_map(data, fft_size=4):
    shape = data.shape
    size = list(shape)
    size[1] = fft_size
    new_data = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        new_data[i, :, :, :] = radar_cube_map_2d(data[i], fft_size)
    return new_data



def radar_cube_map_2d(data, fft_size=4):
    data = cp.asarray(data)
    data = cp.fft.fft(data, axis=1)  # Range FFT
    data = cp.fft.fft(data, axis=2)  # Angle FFT
    data = cp.fft.fft(data, n=fft_size, axis=0)  # Velocity FFT
    data = cp.abs(data)
    data = cp.asnumpy(data)
    return data






