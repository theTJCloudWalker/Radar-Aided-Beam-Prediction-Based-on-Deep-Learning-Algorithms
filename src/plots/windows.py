import numpy as np
import scipy
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



from src.data import data_loader

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

    chirp = adc_data[0, :, 0, 0]

    print(chirp.shape)

