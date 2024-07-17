import torch
import numpy as np
from PIL import Image
from einops import rearrange


def pil2tensor(image: Image) -> torch.Tensor:
    tensor = torch.Tensor(np.array(image).astype(np.float32) / 255.0)
    tensor = rearrange(tensor, 'h w c -> 1 c h w')
    return tensor


def tensor2pil(tensor: torch.Tensor) -> Image:
    tensor = rearrange(tensor.numpy(), '1 c h w -> h w c')
    tensor_array = np.uint8(255 * tensor)
    return Image.fromarray(tensor_array)
