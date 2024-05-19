import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from models.meta_layers import *
import collections

from torch.utils.checkpoint import checkpoint


class MyClassifier(nn.Module):
    def zero_one_loss(self, h, t, is_logistic=False):
        self.eval()
        positive = 1
        negative = 0 if is_logistic else -1

        n_p = (t == positive).sum()
        n_n = (t == negative).sum()
        size = n_p + n_n

        n_pp = (h == positive).sum()
        t_p = ((h == positive) * (t == positive)).sum()
        t_n = ((h == negative) * (t == negative)).sum()
        f_p = n_n - t_n
        f_n = n_p - t_p

        presicion = 0.0 if t_p == 0 else t_p / (t_p + f_p)
        recall = 0.0 if t_p == 0 else t_p / (t_p + f_n)

        return presicion, recall, 1 - (t_p + t_n) / size, n_pp

    def error(self, DataLoader, is_logistic=False):
        targets_all = np.array([])
        prediction_all = np.array([])
        self.eval()
        for data, _, target in DataLoader:
            data = data.cuda()
            t = target.detach().cpu().numpy()
            size = len(t)
            if is_logistic:
                h = np.reshape(torch.sigmoid(self(data)).detach().cpu().numpy(), size)
                h = np.where(h > 0.5, 1, 0).astype(np.int32)
            else:
                h = np.reshape(torch.sign(self(data)).detach().cpu().numpy(), size)

            targets_all = np.hstack((targets_all, t))
            prediction_all = np.hstack((prediction_all, h))

        return self.zero_one_loss(prediction_all, targets_all, is_logistic)

    def evalution_with_density(self, DataLoader, prior):
        targets_all = np.array([])
        prediction_all = np.array([])
        self.eval()
        for data, target in DataLoader:
            data = data.to(device)
            t = target.detach().cpu().numpy()
            size = len(t)
            h = np.reshape(self(data).detach().cpu().numpy(), size)
            h = self.predict_with_density_threshold(h, target, prior)

            targets_all = np.hstack((targets_all, t))
            prediction_all = np.hstack((prediction_all, h))

        return self.zero_one_loss(prediction_all, targets_all)

    def predict_with_density_threshold(self, f_x, target, prior):
        density_ratio = f_x / prior
        sorted_density_ratio = np.sort(density_ratio)
        size = len(density_ratio)

        n_pi = int(size * prior)
        threshold = (
            sorted_density_ratio[size - n_pi] + sorted_density_ratio[size - n_pi - 1]
        ) / 2
        h = np.sign(density_ratio - threshold).astype(np.int32)
        return h


class LeNet(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(LeNet, self).__init__()

        self.input_dim = dim

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)

        self.layer1 = nn.Sequential(self.conv1, self.mp, self.relu)
        self.layer2 = nn.Sequential(self.conv2, self.mp, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu)

        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3])

        self.layer4 = nn.Sequential(self.fc1, self.bn_fc1, self.relu)
        self.classifier = nn.Linear(84, 1)

    def forward(self, x):
        h = x
        for i, layer_module in enumerate(self.layers):
            h = layer_module(h)

        h = h.view(h.size(0), -1)
        h = self.layer4(h)
        h = self.classifier(h)
        return h


class MixLeNet(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(MixLeNet, self).__init__()

        self.input_dim = dim

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)

        self.layer1 = nn.Sequential(self.conv1, self.mp, self.relu)
        self.layer2 = nn.Sequential(self.conv2, self.mp, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu)

        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3])

        self.layer4 = nn.Sequential(self.fc1, self.bn_fc1, self.relu)
        self.classifier = nn.Linear(84, 1)

    def forward(self, x, x2=None, l=None, mix_layer=1000, flag_feature=False):
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)

                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        h_ = h.view(h.size(0), -1)
        h_ = self.layer4(h_)
        h = self.classifier(h_)

        if flag_feature:
            return h, h_
        else:
            return h


class CNNSTL(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(CNNSTL, self).__init__()

        self.input_dim = dim

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.mp = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)

        self.layer1 = nn.Sequential(self.conv1, self.relu, self.mp)
        self.layer2 = nn.Sequential(self.conv2, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu, self.mp)
        self.layer4 = nn.Sequential(self.conv4, self.relu, self.mp)

        self.layers = nn.ModuleList(
            [self.layer1, self.layer2, self.layer3, self.layer4]
        )

        self.layer5 = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu)

        self.classifier = nn.Linear(84, 1)

    def forward(self, x):
        h = x
        for i, layer_module in enumerate(self.layers):
            h = layer_module(h)

        h = h.view(h.size(0), -1)
        h = self.layer5(h)
        h = self.classifier(h)
        return h


class MixCNNSTL(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(MixCNNSTL, self).__init__()

        self.input_dim = dim

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.mp = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)

        self.layer1 = nn.Sequential(self.conv1, self.relu, self.mp)
        self.layer2 = nn.Sequential(self.conv2, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu, self.mp)
        self.layer4 = nn.Sequential(self.conv4, self.relu, self.mp)

        self.layers = nn.ModuleList(
            [self.layer1, self.layer2, self.layer3, self.layer4]
        )

        self.layer5 = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu)

        self.classifier = nn.Linear(84, 1)

    def forward(self, x, x2=None, l=None, mix_layer=1000, flag_feature=False):
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)

                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        h_ = h.view(h.size(0), -1)
        h_ = self.layer5(h_)
        h = self.classifier(h_)

        if flag_feature:
            return h, h_
        else:
            return h


class CNNCIFAR(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(CNNCIFAR, self).__init__()

        self.af = F.relu
        self.input_dim = dim

        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 1)
        self.conv4 = nn.Conv2d(192, 10, 1)
        self.fc1 = nn.Linear(1960, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.af(h)
        h = self.conv2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.af(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h


class MixCNNCIFAR(MyClassifier, MetaModule):
    def __init__(self, dim):
        super(MixCNNCIFAR, self).__init__()

        self.af = nn.ReLU()
        self.input_dim = dim

        self.conv_list = [
            MetaConv2d(3, 96, 3),
            MetaConv2d(96, 96, 3, stride=2),
            MetaConv2d(96, 192, 1),
            MetaConv2d(192, 10, 1),
        ]
        self.fc1 = MetaLinear(1960, 1000)
        self.fc2 = MetaLinear(1000, 1000)
        self.fc3 = MetaLinear(1000, 1)

        self.layers = nn.ModuleList(
            [nn.Sequential(self.conv_list[i], self.af) for i in range(4)]
        )

        self.classifier1 = nn.Sequential(
            self.fc1,
            self.af,
            self.fc2,
            self.af,
        )

    def forward(self, x, x2=None, l=None, mix_layer=1000, flag_feature=False):
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)

                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        h_ = h.view(h.size(0), -1)
        h_ = self.classifier1(h_)
        h = self.fc3(h_)

        if flag_feature:
            return h, h_
        else:
            return h


class MixCNNCIFAR_CL_(MyClassifier, MetaModule):
    def __init__(self, dim):
        super(MixCNNCIFAR_CL_, self).__init__()

        self.af = nn.ReLU()
        self.input_dim = dim

        self.conv_list = [
            MetaConv2d(3, 96, 3),
            MetaConv2d(96, 96, 3, stride=2),
            MetaConv2d(96, 192, 1),
            MetaConv2d(192, 10, 1),
        ]
        self.fc1 = MetaLinear(1960, 1000)
        self.fc2 = MetaLinear(1000, 1000)
        self.classifier = MetaLinear(1000, 1)

        self.layers = nn.ModuleList(
            [nn.Sequential(self.conv_list[i], self.af) for i in range(4)]
        )

        self.mlp = nn.Sequential(
            self.fc1,
            self.af,
            self.fc2,
            self.af,
        )

        self.fc4 = MetaLinear(1000, 1000)
        self.fc5 = MetaLinear(1000, 128)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

    def forward(self, x, x2=None, l=None, mix_layer=1000, flag_feature=False):
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)

                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        h_ = h.view(h.size(0), -1)
        h_feat = self.mlp(h_)
        h = self.classifier(h_feat)
        feat_cl = F.normalize(self.head(h_feat), dim=1)

        if flag_feature:
            return h, feat_cl
        else:
            return h


class MixCNNCIFAR_CL(MyClassifier, MetaModule):
    def __init__(self, dim):
        super(MixCNNCIFAR_CL, self).__init__()

        self.af = nn.ReLU()
        self.input_dim = dim

        self.conv_list = [
            MetaConv2d(3, 96, 3),
            MetaConv2d(96, 96, 3, stride=2),
            MetaConv2d(96, 192, 1),
            MetaConv2d(192, 10, 1),
        ]
        self.fc1 = MetaLinear(21160, 1000)
        self.fc2 = MetaLinear(1000, 1000)
        self.classifier = MetaLinear(1000, 1)

        self.layers = nn.ModuleList(
            [nn.Sequential(self.conv_list[i], self.af) for i in range(4)]
        )

        self.mlp = nn.Sequential(
            self.fc1,
            self.af,
            self.fc2,
            self.af,
        )

        self.fc4 = MetaLinear(1000, 1000)
        self.fc5 = MetaLinear(1000, 128)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

    def forward(self, x, x2=None, l=None, mix_layer=1000, flag_feature=False):
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = layer_module(h)

                if h2 is not None:
                    h2 = layer_module(h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = layer_module(h)

        h_ = h.view(h.size(0), -1)
        h_feat = self.mlp(h_)
        h = self.classifier(h_feat)
        feat_cl = F.normalize(self.head(h_feat), dim=1)

        if flag_feature:
            return h, feat_cl
        else:
            return h


class MixCNNSTL_CL(MyClassifier, MetaModule):
    def __init__(self, dim):
        super(MixCNNSTL_CL, self).__init__()

        self.af = nn.ReLU()
        self.input_dim = dim

        self.conv_list = [
            MetaConv2d(3, 96, 3),
            MetaConv2d(96, 96, 3, stride=2),
            MetaConv2d(96, 192, 1),
            MetaConv2d(192, 10, 1),
        ]
        self.fc1 = MetaLinear(21160, 1000)
        self.fc2 = MetaLinear(1000, 1000)
        self.classifier = MetaLinear(1000, 1)

        self.layers = nn.ModuleList(
            [nn.Sequential(self.conv_list[i], self.af) for i in range(4)]
        )

        self.mlp = nn.Sequential(
            self.fc1,
            self.af,
            self.fc2,
            self.af,
        )

        self.fc4 = MetaLinear(1000, 1000)
        self.fc5 = MetaLinear(1000, 128)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

    def forward(self, x, x2=None, l=None, mix_layer=1000, flag_feature=False):
        h, h2 = x, x2
        if mix_layer == -1:
            if h2 is not None:
                h = l * h + (1.0 - l) * h2

        for i, layer_module in enumerate(self.layers):
            if i <= mix_layer:
                h = checkpoint(layer_module, h)

                if h2 is not None:
                    h2 = checkpoint(layer_module, h2)

            if i == mix_layer:
                if h2 is not None:
                    h = l * h + (1.0 - l) * h2

            if i > mix_layer:
                h = checkpoint(layer_module, h)

        h_ = h.view(h.size(0), -1)
        h_feat = self.mlp(h_)
        h = self.classifier(h_feat)
        feat_cl = F.normalize(self.head(h_feat), dim=1)

        if flag_feature:
            return h, feat_cl
        else:
            return h


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MetaCNN(MetaModule):
    def __init__(self, use_checkpoint=False):
        super(MetaCNN, self).__init__()
        self.conv1 = MetaConv2d(3, 96, kernel_size=3, padding=1)
        self.bn1 = MetaBatchNorm2d(96)
        self.conv2 = MetaConv2d(96, 96, kernel_size=3, padding=1)
        self.bn2 = MetaBatchNorm2d(96)
        self.conv3 = MetaConv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.bn3 = MetaBatchNorm2d(96)
        self.conv4 = MetaConv2d(96, 192, kernel_size=3, padding=1)
        self.bn4 = MetaBatchNorm2d(192)
        self.conv5 = MetaConv2d(192, 192, kernel_size=3, padding=1)
        self.bn5 = MetaBatchNorm2d(192)
        self.conv6 = MetaConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.bn6 = MetaBatchNorm2d(192)
        self.conv7 = MetaConv2d(192, 192, kernel_size=3, padding=1)
        self.bn7 = MetaBatchNorm2d(192)
        self.conv8 = MetaConv2d(192, 192, kernel_size=1)
        self.bn8 = MetaBatchNorm2d(192)
        self.conv9 = MetaConv2d(192, 10, kernel_size=1)
        self.bn9 = MetaBatchNorm2d(10)

        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            self.conv2,
            self.bn2,
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            self.conv3,
            self.bn3,
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            self.conv4,
            self.bn4,
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            self.conv5,
            self.bn5,
            nn.ReLU(),
        )

        self.layer6 = nn.Sequential(
            self.conv6,
            self.bn6,
            nn.ReLU(),
        )

        self.layer7 = nn.Sequential(
            self.conv7,
            self.bn7,
            nn.ReLU(),
        )

        self.layer8 = nn.Sequential(
            self.conv8,
            self.bn8,
            nn.ReLU(),
        )

        self.layer9 = nn.Sequential(
            self.conv9,
            self.bn9,
            nn.ReLU(),
        )

        self.l1 = MetaLinear(640, 1000)
        self.l2 = MetaLinear(1000, 1000)
        self.classifier = MetaLinear(1000, 1)

        self.fc4 = MetaLinear(1000, 1000)
        self.fc5 = MetaLinear(1000, 128)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

        self.apply(weights_init)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, flag_feature=False):
        out = x
        out = out + torch.zeros(
            1, dtype=out.dtype, device=out.device, requires_grad=True
        )

        if self.use_checkpoint:
            out = checkpoint(self.layer1, out)
            out = checkpoint(self.layer2, out)
            out = checkpoint(self.layer3, out)
            out = checkpoint(self.layer4, out)
            out = checkpoint(self.layer5, out)
            out = checkpoint(self.layer6, out)
            out = checkpoint(self.layer7, out)
            out = checkpoint(self.layer8, out)
            out = checkpoint(self.layer9, out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)
            out = self.layer8(out)
            out = self.layer9(out)

        out = out.view(-1, 640)
        out = self.l1(out)
        out = F.relu(out)
        out = self.l2(out)
        h_feat = F.relu(out)

        h = self.classifier(h_feat)

        feat_cl = F.normalize(self.head(h_feat), dim=1)

        if flag_feature:
            return h, feat_cl
        else:
            return h
