import importlib
import time
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F


class ModelAdapter:
    def __init__(self):
        self.use_GPU = torch.cuda.is_available()
        self.model = None
        self.model_config = None

    def load_model(self, model_config: Dict) -> None:
        assert 'class_name' in model_config
        assert 'pretrained_path' in model_config
        self.model_config = model_config
        # 动态导入模型类
        module_name = f"model.{model_config['class_name']}"
        class_name = model_config['class_name']
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        init_params = model_config.get('init_params', {})

        # 实例化模型
        model = model_class(**init_params)
        if self.use_GPU:
            model = model.cuda()
        self.model = model

        # 加载预训练模型
        checkpoint_path = model_config['pretrained_path']
        device = 'cuda' if self.use_GPU else 'cpu'
        ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
        state_dict_mapping = model_config.get('state_dict_mapping', None)
        if state_dict_mapping is not None:
            ckpt = ckpt[state_dict_mapping]

        # 兼容多 GPU 模型
        if next(iter(ckpt)).startswith('module.'):
            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                name = k[7:]
                new_ckpt[name] = v
            ckpt = new_ckpt
        self.model.load_state_dict(ckpt)

        self.model.eval()

    def run_forward(self, input_data: torch.Tensor, clamp=True):
        # pad 到 resize_factor 的整数倍
        resize_factor = self.model_config.get('resize_factor', None)
        h, w = input_data.shape[-2:]
        if resize_factor is not None:
            new_h = (h + resize_factor - 1) // resize_factor * resize_factor
            new_w = (w + resize_factor - 1) // resize_factor * resize_factor
            input_data = F.pad(input_data, (0, new_w - w, 0, new_h - h), mode='reflect')
        with torch.no_grad():
            if self.use_GPU:
                input_data = input_data.cuda()
            start_time = time.time()

            out_data = self.model(input_data)
            if clamp:
                out_data = torch.clamp(out_data, 0., 1.)

            end_time = time.time()
            dur_time = end_time - start_time

            if self.use_GPU:
                out_data = out_data.cpu()
        if resize_factor is not None:
            out_data = out_data[..., :h, :w]

        return out_data, dur_time
