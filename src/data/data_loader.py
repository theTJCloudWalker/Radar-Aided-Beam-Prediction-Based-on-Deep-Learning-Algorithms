import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from ..utils import config_parser


def load_radar_data(root_dir, scenario_id):
    dataset = config_parser.parse_dataset(root_dir)
    scenarios = dataset["scenarios"]
    for scenario_config in scenarios:
        if scenario_config["id"] == scenario_id:
            length = scenario_config["length"]
            shape = scenario_config["shape"]
            X = np.zeros((length,) + tuple(shape), dtype='complex_')
            csv_file = pd.read_csv(os.path.join(scenario_config["path"], scenario_config["csv_filename"]))
            radar_column = scenario_config["radar_column"]
            if scenario_config["type"] == "mat":
                for i in tqdm(range(length)):
                    X[i] = loadmat(os.path.join(scenario_config["path"], csv_file[radar_column][i]))['data']
            elif scenario_config["type"] == "npy":
                for i in tqdm(range(length)):
                    X[i] = np.load(os.path.join(scenario_config["path"], csv_file[radar_column][i]))


            Y = np.array(csv_file[scenario_config["label_column"]])
            return X, Y

    return None
