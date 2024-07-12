import numpy as np
from tqdm import tqdm


def range_angle_map(data, fft_size=64):
    shape = data.shape
    #shape:[length, RX, samples, chirps]
    size = list(shape)
    size[1] = 1
    size[2] = fft_size
    size[3] = 256
    result = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(shape[0])):
        result[i, 0, :, :] = range_angle_map_2d(data[i], fft_size)
    return result


def ca_cfar(signal, num_guard_cells, num_train_cells, rate_fa):
    """
    Cell-Averaging CFAR (CA-CFAR) algorithm for 2D radar signal.

    Parameters:
    signal -- 2D radar signal (Range-Doppler Map)
    num_guard_cells -- Number of guard cells
    num_train_cells -- Number of training cells
    rate_fa -- Desired probability of false alarm

    Returns:
    detection_map -- 2D binary map indicating detected targets
    """
    num_rows, num_cols = signal.shape
    detection_map = np.zeros(signal.shape, dtype=int)

    alpha = num_train_cells * (rate_fa ** (-1 / num_train_cells) - 1)

    for row in range(num_guard_cells + num_train_cells, num_rows - num_guard_cells - num_train_cells):
        for col in range(num_guard_cells + num_train_cells, num_cols - num_guard_cells - num_train_cells):
            sum_train_cells = 0

            for i in range(-num_guard_cells - num_train_cells, num_guard_cells + num_train_cells + 1):
                for j in range(-num_guard_cells - num_train_cells, num_guard_cells + num_train_cells + 1):
                    if abs(i) > num_guard_cells or abs(j) > num_guard_cells:
                        sum_train_cells += signal[row + i, col + j]

            noise_level = sum_train_cells / (num_train_cells * num_train_cells)
            threshold = alpha * noise_level

            if signal[row, col] > threshold:
                detection_map[row, col] = 1

    return detection_map



def range_angle_map_2d(data, fft_size=64):
    # shape:[RX, samples, chirps]
    data = np.fft.fft(data, axis=1)  # Range FFT
    # 突出特征
    data -= np.mean(data, axis=2, keepdims=True)
    data = np.fft.fft(data, n=fft_size, axis=0)  # Angle FFT
    data = np.abs(data).sum(axis=2)  # Sum over velocity
    return data