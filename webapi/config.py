import json
from typing import List, Dict
from glob import glob
import os


def parse_config() -> List[Dict]:
    config_file_list = glob('./config/*.json')
    config_list = []
    for config_file in config_file_list:
        with open(config_file, 'r') as f:
            configs = json.load(f)
            for config in configs:
                pretrained_path = config['pretrained_path']
                if not os.path.exists(pretrained_path):
                    print(f"Pretrained model not found: {pretrained_path}. This model will be ignored.")
                    continue
                config_list.append(config)
    print(f"Loaded {len(config_list)} models.")
    return config_list
