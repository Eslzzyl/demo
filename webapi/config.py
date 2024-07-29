import json
from typing import List, Dict
from glob import glob
import os


def parse_config() -> List[Dict]:
    """
    解析配置文件
    """
    # 列出指定目录下的所有 json 文件
    config_file_list = glob('./config/*.json')
    config_list = []
    for config_file in config_file_list:
        with open(config_file, 'r') as f:
            configs = json.load(f)
            for config in configs:
                # 检查模型的权重文件是否存在。如果不存在，则输出提示信息并直接跳过这个模型
                pretrained_path = config['pretrained_path']
                if not os.path.exists(pretrained_path):
                    print(f"Pretrained model not found: {pretrained_path}. This model will be ignored.")
                    continue
                config_list.append(config)
    # 最后输出总共加载了几个模型的配置信息
    print(f"Loaded {len(config_list)} models.")
    return config_list
