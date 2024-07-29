import torch
import torch.nn as nn
import torch.nn.functional as F


class PReNet(nn.Module):
    """
    PReNet

    与 PRN 相比，带了一个 LSTM

    模型中的 5 个 residual blocks 具有独立的权重参数
    """

    def __init__(self, recurrent_iter=6):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = torch.cuda.is_available()

        # 论文中提到的 f_{in}
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        # 以下的 5 个 res_conv 是论文中提到的 residual blocks
        # 在不带 r 的 PReNet 中，5 个 residual blocks 具有独立的权重参数
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = torch.zeros(batch_size, 32, row, col)
        c = torch.zeros(batch_size, 32, row, col)

        if self.use_GPU:
            last_gpu_index = torch.cuda.device_count() - 1
            device = torch.device(f"cuda:{last_gpu_index}")
            h = h.to(device)
            c = c.to(device)

        # x_list = []
        for i in range(self.iteration):
            # 将上一轮结果 x_t 和带雨图像 y 拼接
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            # x_list.append(x)

        # return x, x_list
        return x
