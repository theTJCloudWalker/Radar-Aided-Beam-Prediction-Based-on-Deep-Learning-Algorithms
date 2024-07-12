import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data import data_loader, preprocessing_torch, preprocessing_cupy

project_root_dir = "C:\\Users\\cloudwalker\\OneDrive - 80shgy\\last dance\\ML-based-beam-prediction"

scenario_id = 32
print("Starting load data : scenario", scenario_id)
X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)
print(X.shape)


idx = 81
radar_data = X[idx, :, :, :]

index = {
    '9': 0,
    '32': 1,
    '33': 1
}

RADAR_PARAMS = [
    {
        'chirps':            128, # number of chirps per frame
        'tx':                  1, # transmitter antenna elements
        'rx':                  4, # receiver antenna elements
        'samples':           256, # number of samples per chirp
        'adc_sampling':      5e6, # Sampling rate [Hz]
        'chirp_slope': 15.015e12, # Ramp (freq. sweep) slope [Hz/s]
        'start_freq':       77e9, # [Hz]
        'idle_time':           5, # Pause between ramps [us]
        'ramp_end_time':      60# Ramp duration [us]
    },
    {
        'chirps':            250, # number of chirps per frame
        'tx':                  1, # transmitter antenna elements
        'rx':                  4, # receiver antenna elements
        'samples':           256, # number of samples per chirp
        'adc_sampling':      6.2e6, # Sampling rate [Hz]
        'chirp_slope': 8.014e12, # Ramp (freq. sweep) slope [Hz/s]
        'start_freq':       77e9, # [Hz]
        'idle_time':           2, # Pause between ramps [us]
        'ramp_end_time':      47.5# Ramp duration [us]
    }
]

i = index[str(scenario_id)]
samples_per_chirp = RADAR_PARAMS[i]['samples']
n_chirps_per_frame = RADAR_PARAMS[i]['chirps']
C = 3e8
chirp_period = (RADAR_PARAMS[i]['ramp_end_time'] + RADAR_PARAMS[i]['idle_time']) * 1e-6

RANGE_RES = ((C * RADAR_PARAMS[i]['adc_sampling']) /
                    (2*RADAR_PARAMS[i]['samples'] * RADAR_PARAMS[i]['chirp_slope']))

VEL_RES_KMPH = 3.6 * C / (2 * RADAR_PARAMS[i]['start_freq'] *
                          chirp_period * RADAR_PARAMS[i]['chirps'])

min_range_to_plot = 0
max_range_to_plot = 250 # m
# set range variables
acquired_range = samples_per_chirp * RANGE_RES
first_range_sample = np.ceil(samples_per_chirp * min_range_to_plot /
                            acquired_range).astype(int)
last_range_sample = np.ceil(samples_per_chirp * max_range_to_plot /
                            acquired_range).astype(int)
round_min_range = first_range_sample / samples_per_chirp * acquired_range
round_max_range = last_range_sample / samples_per_chirp * acquired_range

# Range-Velocity Plot
vel = 125 # Set velocity range as 0 to 125
ang_lim = 75 # comes from array dimensions and frequencies


def minmax(arr):
    return (arr - arr.min())/ (arr.max()-arr.min())


def range_velocity_map(data, fft_size = 128):
    data = np.fft.fft(data, axis=1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, n=fft_size, axis=2) # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis = 0) # Sum over antennas
    data = np.log(1+data)
    return data


def range_angle_map(data, fft_size = 1024):
    data = np.fft.fft(data, axis = 1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis = 0) # Angle FFT
    data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis = 2) # Sum over velocity
    return data.T


def range_velocity_angle(data, fft_size = 4):
    data = np.fft.fft(data, axis=1) # Range FFT
    data = np.fft.fft(data, axis=2) # Velocity FFT
    data = np.fft.fft(data, n=fft_size, axis=0) # Angle FFT
    data = np.abs(data)
    return data

# 绘制图像
fig, axs = plt.subplots(figsize=(12, 6), ncols=2, tight_layout=True)

# # Range-Angle Plot
# radar_range_ang_data = range_angle_map(radar_data, 512)[first_range_sample:last_range_sample]
# axs[0].imshow(minmax(radar_range_ang_data), aspect='auto',
#               extent=[-ang_lim, +ang_lim, round_min_range, round_max_range],
#               cmap='seismic', origin='lower')
# axs[0].set_xlabel('Angle Bin')
# axs[0].set_ylabel('Range Bin')
# axs[0].set_title('Range-Angle Map')
#
# # Range-Doppler Plot
# radar_range_dop_data = range_velocity_map(radar_data, 512)[first_range_sample:last_range_sample]
# axs[1].imshow(minmax(radar_range_dop_data), aspect='auto',
#               extent=[-vel, +vel, round_min_range, round_max_range],
#               cmap='seismic', origin='lower')
# axs[1].set_xlabel('Doppler Bin')
# axs[1].set_ylabel('Range Bin')
# axs[1].set_title('Range-Doppler Map')
#
# plt.show()

# Range-Angle Plot
radar_range_ang_data = range_angle_map(radar_data, 2048)[first_range_sample:last_range_sample]
axs[0].imshow(radar_range_ang_data, aspect='auto',
              extent=[-ang_lim, +ang_lim, round_min_range, round_max_range],
              cmap='seismic', origin='lower')
axs[0].set_xlabel('Angle Bin')
axs[0].set_ylabel('Range Bin')
axs[0].set_title('Range-Angle Map')

# Range-Doppler Plot
radar_range_dop_data = range_velocity_map(radar_data, 2048)[first_range_sample:last_range_sample]
axs[1].imshow(radar_range_dop_data, aspect='auto',
              extent=[-vel, +vel, round_min_range, round_max_range],
              cmap='seismic', origin='lower')
axs[1].set_xlabel('Doppler Bin')
axs[1].set_ylabel('Range Bin')
axs[1].set_title('Range-Doppler Map')

plt.show()

