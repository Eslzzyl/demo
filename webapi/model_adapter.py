import importlib
import time
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F


class ModelAdapter:
    """
    用于适配不同的图像处理模型
    """
    def __init__(self):
        self.use_GPU = torch.cuda.is_available()
        self.model = None
        self.model_config = None
        # 使用最后一张GPU
        if self.use_GPU:
            # 确定最后一个GPU的索引
            last_gpu_index = torch.cuda.device_count() - 1
            self.device = torch.device(f"cuda:{last_gpu_index}")
        else:
            self.device = torch.device('cpu')
        print(self.device)

    async def load_model(self, model_config: Dict) -> None:
        """
        通过给定的模型配置信息，初始化对应的模型并加载预训练权重。
        """
        # 以下两个 assert 用于确保对应的字段存在，因为它们是必不可少的
        assert 'class_name' in model_config
        assert 'pretrained_path' in model_config
        self.model_config = model_config
        # 动态导入模型类
        module_name = f"model.{model_config['class_name']}"
        class_name = model_config['class_name']
        # 此处约定模型的 module 和 class_name 相同，先导入 module
        module = importlib.import_module(module_name)
        # 再导入主类
        model_class = getattr(module, class_name)
        # 获取模型主类在初始化时的参数。这里的参数一般与模型结构有关，例如有几个 Transformer Block 等
        init_params = model_config.get('init_params', {})

        # 实例化模型，同时传入初始化参数，然后将模型移动到指定设备
        self.model = model_class(**init_params).to(self.device)

        # 加载预训练模型
        checkpoint_path = model_config['pretrained_path']
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # 有些模型会在 chekpoint 中使用一个单独的字段保存模型的权重，此处用于进行这个适配
        # 例如，ckpt['model']，ckpt['params'] 等
        state_dict_mapping = model_config.get('state_dict_mapping', None)
        # 如果在配置文件中确实配置了这种单独一个字段存储权重，那么就用这个字段覆盖 ckpt
        if state_dict_mapping is not None:
            ckpt = ckpt[state_dict_mapping]

        # 兼容多 GPU 模型。使用 torch.DataParallel 训练的模型，保存时模型的所有参数都会加一个额外的 module. 前缀
        # 这里首先检测是否存在这个前缀，如果有，则去除它。
        if next(iter(ckpt)).startswith('module.'):
            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                name = k[7:]
                new_ckpt[name] = v
            ckpt = new_ckpt
        # 加载权重
        self.model.load_state_dict(ckpt)

        # 进入评估模式，准备进行推理
        self.model.eval()

    def run_forward(self, input_data: torch.Tensor, clamp=True):
        """
        执行 forward 推理
        """
        # 有些图像处理模型要求输入图像的长和宽都必须是某个数字（如 8 或 16）的整数倍，此处用于进行这一处理
        # pad 到 resize_factor 的整数倍
        resize_factor = self.model_config.get('resize_factor', None)
        h, w = input_data.shape[-2:]
        if resize_factor is not None:
            new_h = (h + resize_factor - 1) // resize_factor * resize_factor
            new_w = (w + resize_factor - 1) // resize_factor * resize_factor
            input_data = F.pad(input_data, (0, new_w - w, 0, new_h - h), mode='reflect')
        # 开始推理
        with torch.no_grad():
            # 将输入数据移动到指定设备
            if self.use_GPU:
                input_data = input_data.to(self.device)
            # 计时
            start_time = time.time()

            # 使数据通过模型
            out_data = self.model(input_data)
            # 如果指定了 clamp 操作（默认开启），则进行一次 clamp
            if clamp:
                out_data = torch.clamp(out_data, 0., 1.)

            # 计时
            end_time = time.time()
            dur_time = end_time - start_time

            # 必要时，将数据移动回 CPU
            if self.use_GPU:
                out_data = out_data.cpu()
        # 如果先前对输入数据进行了 resize，那么此时要截取数据，保证输入和输出的大小是一致的。
        if resize_factor is not None:
            out_data = out_data[..., :h, :w]

        return out_data, dur_time
