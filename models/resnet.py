import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from models.meta_layers import *

from torch.utils.checkpoint import checkpoint


class PreActBlockMeta(MetaModule):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockMeta, self).__init__()
        self.bn1 = MetaBatchNorm2d(in_planes)
        self.conv1 = MetaConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneckMeta(MetaModule):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckMeta, self).__init__()
        self.bn1 = MetaBatchNorm2d(in_planes)
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNetMeta(MetaModule):
    def __init__(self, block, num_blocks, num_classes=10, use_checkpoint=False):
        super(PreActResNetMeta, self).__init__()
        self.in_planes = 64

        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        input_dim = 512 * block.expansion
        self.classifier = MetaLinear(input_dim, num_classes)

        self.fc4 = MetaLinear(input_dim, input_dim)
        self.fc5 = MetaLinear(input_dim, 128)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

        self.use_checkpoint = use_checkpoint

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, flag_feature=False):
        out = x
        out = out + torch.zeros(
            1, dtype=out.dtype, device=out.device, requires_grad=True
        )
        if self.use_checkpoint:
            out = checkpoint(self.conv1, out)
            out = checkpoint(self.layer1, out)
            out = checkpoint(self.layer2, out)
            out = checkpoint(self.layer3, out)
            out = checkpoint(self.layer4, out)
        else:
            out = self.conv1(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        out = F.adaptive_max_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        y = self.classifier(out)
        feat_cl = F.normalize(self.head(out), dim=1)
        if flag_feature:
            return y, feat_cl
        else:
            return y


def preact_resnet_meta18():
    return PreActResNetMeta(PreActBlockMeta, [2, 2, 2, 2])


def preact_resnet_meta2332():
    return PreActResNetMeta(PreActBlockMeta, [2, 3, 3, 2])


def preact_resnet_meta3333():
    return PreActResNetMeta(PreActBlockMeta, [3, 3, 3, 3])


def preact_resnet_meta34():
    return PreActResNetMeta(PreActBlockMeta, [3, 4, 6, 3])


def preact_resnet_meta50():
    return PreActResNetMeta(PreActBottleneckMeta, [3, 4, 6, 3])


def preActResNetMeta101():
    return PreActResNetMeta(PreActBottleneckMeta, [3, 4, 23, 3])


def preActResNetMeta152():
    return PreActResNetMeta(PreActBottleneckMeta, [3, 8, 36, 3])
