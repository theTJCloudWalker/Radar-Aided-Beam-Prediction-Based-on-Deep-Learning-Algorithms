import numpy as np
import matplotlib.pyplot as plt

from src.data import data_loader, preprocessing
from src.plots import process


if __name__ == '__main__':
    project_root_dir = "C:\\Users\\cloudwalker\\OneDrive - 80shgy\\last dance\\ML-based-beam-prediction"

    scenario_id = 35
    print("Starting load data : scenario", scenario_id)
    X, Y = data_loader.load_radar_data(project_root_dir, scenario_id)

    adc_data = X
    print(adc_data.shape)
    # [length, rx, samples, chirps]
    print(adc_data.dtype)
    adc_data = adc_data.transpose(0, 2, 1, 3)

    range_image, range_doppler_image, rda_cube = process.get_range_doppler_angle(adc_data[0, :, :, :])
    print(range_doppler_image.shape)
    process.show_3_images([np.absolute(range_image.mean(axis=1)), 'range doppler', 'Doppler', 'Range'],
                  [np.absolute(rda_cube.mean(axis=2)), 'range angle', 'Angle', 'Range'],
                  [np.absolute(rda_cube.mean(axis=0)), 'angle doppler', 'Doppler', 'Angle']
                  )
    range_doppler_image = preprocessing.ca_cfar(range_doppler_image, 2, 5,1e-3)
    process.show_3_images([np.absolute(range_doppler_image), 'range doppler', 'Doppler', 'Range'],
                  [np.absolute(rda_cube.mean(axis=2)), 'range angle', 'Angle', 'Range'],
                  [np.absolute(rda_cube.mean(axis=0)), 'angle doppler', 'Doppler', 'Angle']
                  )

