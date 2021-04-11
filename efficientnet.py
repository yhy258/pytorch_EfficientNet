import torch
import torch.nn as nn
from sub_models import SepConv, MBConv, Swish

class EfficientNet(nn.Module):
    def __init__(self, num_classes, width_coef=1.0, height_coef=1.0, resolution_coef=1.0, dropout=0.2, se_scale=4,
                 stochastic_depth=False, p=0.5):
        super().__init__()

        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]  # width
        repeats = [1, 2, 2, 3, 3, 4, 1]  # depth

        self.channels = [int(i * width_coef) for i in channels]
        self.repeats = [int(i * height_coef) for i in repeats]
        self.strides = [2, 1, 2, 2, 2, 1, 2, 1]

        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(self.repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        self.upsample = nn.Upsample(scale_factor=resolution_coef, mode='bilinear', align_corners=False)

        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=3, stride=self.strides[0], padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            Swish()
        )
        self.stage_2 = self._make_block(SepConv, self.repeats[0], self.channels[0], self.channels[1], 3,
                                        stride=self.strides[1], se_scale=se_scale)
        self.stage_3 = self._make_block(MBConv, self.repeats[1], self.channels[1], self.channels[2], 3,
                                        stride=self.strides[2], se_scale=se_scale)
        self.stage_4 = self._make_block(MBConv, self.repeats[2], self.channels[2], self.channels[3], 5,
                                        stride=self.strides[3], se_scale=se_scale)
        self.stage_5 = self._make_block(MBConv, self.repeats[3], self.channels[3], self.channels[4], 3,
                                        stride=self.strides[4], se_scale=se_scale)
        self.stage_6 = self._make_block(MBConv, self.repeats[4], self.channels[4], self.channels[5], 5,
                                        stride=self.strides[5], se_scale=se_scale)
        self.stage_7 = self._make_block(MBConv, self.repeats[5], self.channels[5], self.channels[6], 5,
                                        stride=self.strides[6], se_scale=se_scale)
        self.stage_8 = self._make_block(MBConv, self.repeats[6], self.channels[6], self.channels[7], 3,
                                        stride=self.strides[7], se_scale=se_scale)
        self.stage_9 = nn.Sequential(
            nn.Conv2d(self.channels[7], self.channels[8], 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.channels[8]),
            Swish(),
        )
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.channels[8], num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.stage_5(x)
        x = self.stage_6(x)
        x = self.stage_7(x)
        x = self.stage_8(x)
        x = self.stage_9(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def _make_block(self, block, repeat, in_channel, out_channel, kernel_size, stride, se_scale):
        block_stride = [stride] + [1] * (repeat - 1)
        layers = []
        for stride in block_stride:
            layers.append(block(in_channel, out_channel, kernel_size, stride, se_scale, self.p))
            in_channel = out_channel
            self.p -= self.step
        return nn.Sequential(*layers)



def make_EfficientNet(num_classes,width, height,scale, dropout,se_scale, stochastic_depth, p):
    return EfficientNet(num_classes, width, height,scale, dropout, se_scale, stochastic_depth, p)