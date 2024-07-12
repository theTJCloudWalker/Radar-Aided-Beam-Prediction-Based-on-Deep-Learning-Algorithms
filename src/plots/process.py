import os
import numpy as np
import scipy
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

from src.data import data_loader


def show_one_chirp(one_chirp_data, y_label='amplitude', x_label='One chirp'):
    plt.figure(figsize=[8, 6])
    plt.plot(one_chirp_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def show_image(image_data, image_name, x_label='', y_label=''):
    plt.imshow(image_data)
    plt.title(image_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def show_3_images(img_data1, img_data2, img_data3):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img_data1[0], aspect=1.44)
    plt.title(img_data1[1])
    plt.xlabel(img_data1[2])
    plt.ylabel(img_data1[3])

    plt.subplot(2, 2, 2)
    plt.imshow(img_data2[0], aspect=1.0)
    plt.title(img_data2[1])
    plt.xlabel(img_data2[2])
    plt.ylabel(img_data2[3])

    plt.subplot(2, 2, 3)
    plt.imshow(img_data3[0], aspect=1.0)
    plt.title(img_data3[1])
    plt.xlabel(img_data3[2])
    plt.ylabel(img_data3[3])
    plt.show()


def get_range_doppler_angle(adc_data_in):
    # adc_data_in - ADC samples, vRx, chirps
    samples_in = adc_data_in.shape[0]
    range_window = np.hamming(samples_in).reshape(-1, 1, 1)
    range_data = np.fft.fft(adc_data_in * range_window, samples_in, axis=0)
    #
    chirps_in = range_data.shape[2]
    doppler_window = np.hamming(chirps_in).reshape(1, 1, -1)
    range_doppler_data = np.fft.fftshift(np.fft.fft(range_data * doppler_window, chirps_in, axis=2), axes=2)
    #
    # samples, vRx, chirps
    angle_window = np.hamming(range_doppler_data.shape[1]).reshape(1, -1, 1)
    angle_bins = 180
    rda_data = np.fft.fftshift(np.fft.fft(range_doppler_data * angle_window, angle_bins, axis=1), axes=1)
    return range_data, range_doppler_data, rda_data


if __name__ == '__main__':
    project_root_dir = "C:\\Users\\cloudwalker\\OneDrive - 80shgy\\last dance\\ML-based-beam-prediction"

    scenario_id = 9
    print("Starting load data : scenario", scenario_id)
    X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)

    adc_data = X
    print(adc_data.shape)
    # [length, rx, samples, chirps]
    print(adc_data.dtype)
    adc_data = adc_data.transpose(0, 2, 1, 3)

    chirp = adc_data[0, :, 0, 0]
    show_one_chirp(np.absolute(chirp), x_label='IF signal of a chirp')
    chirp_fft = np.fft.fft(chirp)
    show_one_chirp(np.absolute(chirp_fft), x_label='IF signal amplitude (range)', y_label='Amplitude')

    # show all chirps
    show_one_chirp(np.absolute(adc_data[0, :, 0, :]), x_label='IF signal of frame chirps')
    frame = adc_data[0, :, :, :]
    show_image(np.absolute(np.fft.fft(frame, axis=0).mean(1)), 'range FFT', x_label='Chirps', y_label='Range')

    range_image, range_doppler_image, rda_cube = get_range_doppler_angle(adc_data[0, :, :, :])

    show_3_images([np.absolute(range_image.mean(axis=1)), 'range doppler', 'Doppler', 'Range'],
                  [np.absolute(rda_cube.mean(axis=2)), 'range angle', 'Angle', 'Range'],
                  [np.absolute(rda_cube.mean(axis=0)), 'angle doppler', 'Doppler', 'Angle']
                  )
