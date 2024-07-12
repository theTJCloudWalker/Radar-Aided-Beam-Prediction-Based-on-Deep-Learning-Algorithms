import json
import os.path


def parse_dataset(root_dir):
    path = os.path.join(root_dir, 'config.json')
    with open(path) as config_file:
        config = json.load(config_file)

    dataset = config["dataset"]
    return dataset



