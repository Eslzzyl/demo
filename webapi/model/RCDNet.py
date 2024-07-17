# A Model-driven Deep Neural Network for Single Image Rain Removal
# https://arxiv.org/abs/2005.01333
# http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F

# rain kernel  C initialized by the Matlab code "init_rain_kernel.m"
rain_kernel = io.loadmat('./model/RCDNet_init_kernel.mat')['C9']  # 3*32*9*9
kernel = torch.FloatTensor(rain_kernel)

# filtering on rainy image for initializing B^(0) and Z^(0), refer to supplementary material(SM)
filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)


class RCDNet(nn.Module):
    def __init__(self):
        super(RCDNet, self).__init__()
        self.S = 17  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process
        self.num_M = 32
        self.num_Z = 32

        # Stepsize
        self.etaM = torch.Tensor([1])  # initialization
        self.etaB = torch.Tensor([5])  # initialization
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaB_S = self.make_eta(self.S, self.etaB)

        # Rain kernel
        self.C0 = nn.Parameter(data=kernel, requires_grad=True)  # used in initialization process
        self.C = nn.Parameter(data=kernel, requires_grad=True)  # self.C (rain kernel) is inter-stage sharing

        # filter for initializing B and Z
        self.C_z_const = filter.expand(self.num_Z, 3, -1, -1)  # size: self.num_Z*3*3*3
        self.C_z = nn.Parameter(self.C_z_const, requires_grad=True)

        # proxNet
        self.proxNet_B_0 = Bnet()  # used in initialization process
        self.proxNet_B_S = self.make_Bnet(self.S)
        self.proxNet_M_S = self.make_Mnet(self.S)
        self.proxNet_B_last_layer = Bnet()  # fine-tune at the last

        # for sparse rain layer
        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def make_Bnet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(Bnet())
        return nn.Sequential(*layers)

    def make_Mnet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(Mnet())
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, x):
        x *= 255
        # save mid-updating results
        ListB = []
        ListR = []

        # initialize B0 and Z0 (M0 =0)
        Z00 = F.conv2d(x, self.C_z, stride=1, padding=1)  # dual variable z
        input_ini = torch.cat((x, Z00), dim=1)
        BZ_ini = self.proxNet_B_0(input_ini)
        B0 = BZ_ini[:, :3, :, :]
        Z0 = BZ_ini[:, 3:, :, :]

        # 1st iteration：Updating B0-->M1
        R_hat = x - B0
        R_hat_cut = F.relu(R_hat - self.tau)  # for sparse rain layer
        Epsilon = F.conv_transpose2d(R_hat_cut, self.C0 / 10, stride=1,
                                     padding=4)  # /10 for controlling the updating speed
        M1 = self.proxNet_M_S[0](Epsilon)
        R = F.conv2d(M1, self.C / 10, stride=1, padding=4)  # /10 for controlling the updating speed

        # 1st iteration: Updating M1-->B1
        B_hat = x - R
        B_mid = (1 - self.etaB_S[0] / 10) * B0 + self.etaB_S[0] / 10 * B_hat
        input_concat = torch.cat((B_mid, Z0), dim=1)
        BZ = self.proxNet_B_S[0](input_concat)
        B1 = BZ[:, :3, :, :]
        Z1 = BZ[:, 3:, :, :]
        ListB.append(B1)
        ListR.append(R)
        B = B1
        Z = Z1
        M = M1
        for i in range(self.iter):
            # M-net
            R_hat = x - B
            Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((R - R_hat), self.C / 10, stride=1, padding=4)
            M = self.proxNet_M_S[i + 1](M - Epsilon)

            # B-net
            R = F.conv2d(M, self.C / 10, stride=1, padding=4)
            ListR.append(R)
            B_hat = x - R
            B_mid = (1 - self.etaB_S[i + 1, :] / 10) * B + self.etaB_S[i + 1, :] / 10 * B_hat
            input_concat = torch.cat((B_mid, Z), dim=1)
            BZ = self.proxNet_B_S[i + 1](input_concat)
            B = BZ[:, :3, :, :]
            Z = BZ[:, 3:, :, :]
            ListB.append(B)
        BZ_adjust = self.proxNet_B_last_layer(BZ)
        B = BZ_adjust[:, :3, :, :]
        ListB.append(B)
        return ListB[-1] / 255


# proxNet_M
class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.channels = 32
        self.T = 4  # the number of resblocks in each proxNet
        self.layer = self.make_resblock(self.T)
        self.tau0 = torch.Tensor([0.5])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparse rain map

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              ))
        return nn.Sequential(*layers)

    def forward(self, x):
        M = x
        for i in range(self.T):
            M = F.relu(M + self.layer[i](M))
        M = F.relu(M - self.tau)
        return M


# proxNet_B
class Bnet(nn.Module):
    def __init__(self):
        super(Bnet, self).__init__()
        self.channels = 32 + 3  # 3 means R,G,B channels for color image
        self.T = 4
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x
        for i in range(self.T):
            B = F.relu(B + self.layer[i](B))
        return B
