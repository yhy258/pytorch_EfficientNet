import torch.nn as nn
import torch
import numpy as np

class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(x)

class SEBlock(nn.Module):
    def __init__(self,in_channel, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.exitation = nn.Sequential(
            nn.Linear(in_channel, in_channel*r),
            Swish(),
            nn.Linear(in_channel*r, in_channel),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.squeeze(x)
        x = x.view(x.size(0),-1)
        x = self.exitation(x)
        x = x.view(x.size(0),x.size(1), 1,1)
        return x


# MBConv의 경우 repeat되면서, stride가 1인 경우에만 shortcut add

class MBConv(nn.Module):
    expand = 6

    def __init__(self, in_channel, out_channel, kernel_size, stride, se_scale=4, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channel == out_channel) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * MBConv.expand, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(in_channel * MBConv.expand),
            Swish(),
            nn.Conv2d(in_channel * MBConv.expand, in_channel * MBConv.expand, kernel_size, 1, padding=kernel_size // 2,
                      bias=False, groups=in_channel * MBConv.expand),
            nn.BatchNorm2d(in_channel * MBConv.expand),
            Swish()
        )

        self.se = SEBlock(in_channel * MBConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channel * MBConv.expand, out_channel, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = (in_channel == out_channel) and (stride == 1)

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)

        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x += x_shortcut

        return x


class SepConv(nn.Module):
    expand = 1

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channel == out_channel) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * SepConv.expand, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False, groups=in_channel * SepConv.expand),
            nn.BatchNorm2d(in_channel * SepConv.expand),
            Swish()
        )

        self.se = SEBlock(in_channel * SepConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channel * SepConv.expand, out_channel, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = (in_channel == out_channel) and (stride == 1)

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)
        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x += x_shortcut
        return x




