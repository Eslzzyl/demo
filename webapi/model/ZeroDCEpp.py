import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDNTem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDNTem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


def enhance(x, x_r):
    x = x + x_r * (torch.pow(x, 2) - x)
    x = x + x_r * (torch.pow(x, 2) - x)
    x = x + x_r * (torch.pow(x, 2) - x)
    enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
    x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
    x = x + x_r * (torch.pow(x, 2) - x)
    x = x + x_r * (torch.pow(x, 2) - x)
    enhance_image = x + x_r * (torch.pow(x, 2) - x)

    return enhance_image


class ZeroDCEpp(nn.Module):

    def __init__(self, scale_factor=12):
        super(ZeroDCEpp, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        # zerodce DWC + p-shared
        self.e_conv1 = CSDNTem(3, number_f)
        self.e_conv2 = CSDNTem(number_f, number_f)
        self.e_conv3 = CSDNTem(number_f, number_f)
        self.e_conv4 = CSDNTem(number_f, number_f)
        self.e_conv5 = CSDNTem(number_f * 2, number_f)
        self.e_conv6 = CSDNTem(number_f * 2, number_f)
        self.e_conv7 = CSDNTem(number_f * 2, 3)

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        enhance_image = enhance(x, x_r)
        return enhance_image