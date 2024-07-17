import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size, bias=True,
            conv=default_conv, norm=None, act=nn.ReLU(inplace=True)):

        super(ResBlock, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                modules.append(norm(n_feats))
            if act and i == 0:
                modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feats=64, kernel_size=5, n_resblocks=19,
                 mean_shift=True):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = n_feats
        self.kernel_size = kernel_size
        self.n_resblocks = n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = 255
        self.mean = self.rgb_range / 2

        modules = [default_conv(self.in_channels, self.n_feats, self.kernel_size)]
        for _ in range(self.n_resblocks):
            modules.append(ResBlock(self.n_feats, self.kernel_size))
        modules.append(default_conv(self.n_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        if self.mean_shift:
            x = x - self.mean

        output = self.body(x)

        if self.mean_shift:
            output = output + self.mean

        return output


class ConvEnd(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(ConvEnd, self).__init__()

        modules = [
            default_conv(in_channels, out_channels, kernel_size),
            nn.PixelShuffle(ratio)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)


class DeepDeblur(nn.Module):
    def __init__(self):
        super(DeepDeblur, self).__init__()

        self.rgb_range = 255
        self.mean = self.rgb_range / 2

        self.n_resblocks = 19
        self.n_feats = 64
        self.kernel_size = 5

        self.n_scales = 3

        self.body_models = nn.ModuleList([
            ResNet(3, 3, mean_shift=False),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, ResNet(6, 3, mean_shift=False))

        self.conv_end_models = nn.ModuleList([])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [ConvEnd(3, 12)]

    def forward(self, input_pyramid):
        input_pyramid *= 255

        scales = range(self.n_scales - 1, -1, -1)  # 0: fine, 2: coarse

        for s in scales:
            input_pyramid[s] -= self.mean

        output_pyramid = [None] * self.n_scales

        input_s = input_pyramid[-1]
        for s in scales:  # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                input_s = torch.cat((input_pyramid[s - 1], up_feat), 1)

        for s in scales:
            output_pyramid[s] += self.mean

        output_pyramid /= 255
        return output_pyramid
