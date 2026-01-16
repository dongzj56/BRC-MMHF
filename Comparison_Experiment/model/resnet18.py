"""
resnet18.py — 3D ResNet-18 backbone

用法示例:
>>> from resnet18 import resnet18
>>> net = resnet18(sample_input_D=120, sample_input_H=144, sample_input_W=120,
                   num_seg_classes=1, shortcut_type='B', no_cuda=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = ["ResNet", "resnet18"]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3 × 3 × 3 卷积"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    """A 类残差下采样: AvgPool + 通道补零"""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class ResNet(nn.Module):
    """3D-ResNet-18 主干，layer3/4 使用扩张卷积保持高分辨率"""

    def __init__(
        self,
        block,
        layers,
        sample_input_D,
        sample_input_H,
        sample_input_W,
        num_seg_classes=1,
        shortcut_type="B",
        no_cuda=False,
    ):
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda

        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4
        )

        # 若只做特征提取，可忽略以下 segmentation 头
        self.conv_seg = nn.Sequential(
            nn.ConvTranspose3d(512, 32, 2, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_seg_classes, 1, bias=False),
        )

        self._init_weights()

    def _make_layer(
        self, block, planes, blocks, shortcut_type, stride=1, dilation=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)
        return x


def resnet18(**kwargs):
    """
    构造 3D-ResNet-18。

    关键参数:
    - sample_input_D/H/W: 输入深度/高/宽 (仅影响座位测试)
    - shortcut_type: "A" | "B"
    - num_seg_classes: 若仅提特征, 可保持 1
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


if __name__ == "__main__":
    # 简单自测
    net = resnet18(
        sample_input_D=120,
        sample_input_H=144,
        sample_input_W=120,
        num_seg_classes=1,
        shortcut_type="B",
        no_cuda=True,
    )
    dummy = torch.randn(1, 1, 120, 144, 120)
    out = net(dummy)
    print("Output shape:", out.shape)
