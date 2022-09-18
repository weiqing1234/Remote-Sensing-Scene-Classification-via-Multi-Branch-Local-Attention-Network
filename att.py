import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from cbam import *
from ELSAM import *
from mca import *
from thop import profile
# from .bam import *

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# ---------------------SE-Block---------------------As follows


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=16):
        super(SqueezeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = self.avg_pool(x).view(batch, channels)
        # out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x


# ---------------------SE-Block---------------------Abrove


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, use_elsam=False, use_cbam=False):
        super(BasicBlock, self).__init__()
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.se = SqueezeBlock(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None
        if use_elsam:
            self.elsam = ELSAM(planes, 2)
        else:
            self.elsam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)
        # out = self.elsam(out)
        if not self.elsam is None:
            w = self.elsam(out)
            out = torch.mul(out, w)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, use_elsam=False, use_cbam=False):
        super(Bottleneck, self).__init__()
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.se = SqueezeBlock(planes * 4)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * 4)
        else:
            self.cbam = None
        if use_elsam:
            self.elsam = ELSAM(planes * 4, 2)
        else:
            self.elsam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)
        # out = self.elsam(out)
        if not self.elsam is None:
            w = self.elsam(out)
            out = torch.mul(out, w)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.network_type = network_type
        # different model config between ImageNet and CIFAR
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # self.elasm = ELSAM(self.inplanes, 16)

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        # self.elsam = ELSAM(256, 16)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        # self.elsam = ELSAM(512 * block.expansion, 16)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_elsam=att_type=='elsam', use_cbam=att_type=='cbam'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_elsam=att_type=='elsam', use_cbam=att_type=='cbam'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.elasm(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        # x = self.elsam(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type):

    assert network_type in ["UCM", "AID", "NWPU-RESISC45", "NWPU-RESISC450.1", "AID0.5", "UCMdataaug",
                            "NWPU-RESISC45dataaug", "NWPU-RESISC45dataaug0.1", "AIDdataaug", "AIDdataaug0.5"], "network type should be UCM or AID / NWPU-RESISC45"
    assert att_type in ["clsam", "elsam", "eclsam", "cbam", "mca"]
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, att_type)
        # model.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)
        # pretext_model = torch.load('resnet50-19c8e357.pth')
        # model_dict = model.state_dict()
        # state_dict = {k : v for k, v in pretext_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # model.load_state_dict(model_dict, strict=False)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, att_type)

    return model


def main():
    x = torch.randn([16, 3, 224, 224])
    model = ResidualNet('NWPU-RESISC45', 50, 1000, 'elsam')
    y= model(x)
    print(model)
    print(get_n_params(model))
    flop, params = profile(model, (x,), verbose=False)
    print(flop, params)

if __name__ == '__main__':
    main()
# model = ResidualNet('NWPU-RESISC45', 50)
# model.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = torch.nn.Linear(model.fc.in_features, 45)
# params_to_update = model.parameters()
# print("Params to learn:")
#
# params_to_update = []
# for name,param in model.named_parameters():
#     if param.requires_grad == True:
#         params_to_update.append(param)
#         print("\t",name)
# print(model)

