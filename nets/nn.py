import os
import math
import torch
import urllib.request
import torch.nn as nn
import torchvision as tv
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch.nn.parameter import Parameter
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['Conv2d']


class Conv2d(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True,
                 padding_mode='zeros', padding_init='gaussian',
                 num_tokens=1, data_crop_size=1, device=None, dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Conv2d, self).__init__()

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}

        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(padding, valid_padding_strings))

            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular', 'trainable'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))

        self.groups = groups
        self.transposed = False
        self.stride = _pair(stride)
        self.output_padding = _pair(0)
        self.in_channels = in_channels
        self.dilation = _pair(dilation)
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.padding_init = padding_init
        self.data_crop_size = data_crop_size
        self.kernel_size = _pair(kernel_size)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.num_tokens = self.padding[0] if isinstance(self.padding, tuple) else self.padding

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)

            if padding == 'same':
                for d, k, i in zip(self.dilation, self.kernel_size,
                                   range(len(self.kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                            total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(torch.empty(
            (self.out_channels, self.in_channels // self.groups, *self.kernel_size), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(self.out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if self.padding_mode == 'trainable':
            self._setup_prompt_pad()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)

            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}')

        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'

        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'

        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'

        if self.groups != 1:
            s += ', groups={groups}'

        if self.bias is None:
            s += ', bias=False'

        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'

        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(Conv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def make_padding_trainable(self):
        self.padding_mode = 'trainable'
        self._setup_prompt_pad()

    def _setup_prompt_pad(self):
        if self.padding_init == "random":
            self.prompt_embeddings_tb = nn.Parameter(
                torch.zeros(1, self.in_channels, 2 * self.num_tokens, self.data_crop_size + 2 * self.num_tokens))

            self.prompt_embeddings_lr = nn.Parameter(
                torch.zeros(1, self.in_channels, self.data_crop_size, 2 * self.num_tokens))

            nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

            self.prompt_norm = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )

        elif self.padding_init == "gaussian":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(1, self.in_channels, 2 * self.num_tokens,
                                                                 self.data_crop_size + 2 * self.num_tokens))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(1, self.in_channels, self.data_crop_size,
                                                                 2 * self.num_tokens))

            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)

            self.prompt_norm = nn.Identity()

        elif self.padding_init == "zero":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(1, self.in_channels, 2 * self.num_tokens,
                                                                 self.data_crop_size + 2 * self.num_tokens))
            self.prompt_embeddings_lr = nn.Parameter(
                torch.zeros(1, self.in_channels, self.data_crop_size, 2 * self.num_tokens))

            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

    def _incorporate_prompt(self, x):
        B = x.shape[0]
        prompt_emb_lr = self.prompt_norm(self.prompt_embeddings_lr).expand(B, -1, -1, -1)
        prompt_emb_tb = self.prompt_norm(self.prompt_embeddings_tb).expand(B, -1, -1, -1)
        x = torch.cat((prompt_emb_lr[:, :, :, :self.num_tokens], x, prompt_emb_lr[:, :, :, self.num_tokens:]), dim=-1)
        x = torch.cat((prompt_emb_tb[:, :, :self.num_tokens, :], x, prompt_emb_tb[:, :, self.num_tokens:, :]), dim=-2)
        return x

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode == 'trainable':
            x = self._incorporate_prompt(input)
            return F.conv2d(x, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        if self.padding_mode != 'zeros' and self.padding_mode != 'trainable':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias,
                            self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):

        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x_2 = self.relu(x)
        x = self.maxpool(x_2)

        x_l1 = self.layer1(x)
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)

        x_l5 = self.avgpool(x_l4)

        return [x_2, x_l1, x_l2, x_l3, x_l4, x_l5]

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
              'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
              'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
              'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
              'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth', }


class GazeNet(torch.nn.Module):
    def __init__(self, args):
        super(GazeNet, self).__init__()
        self.gaze_network = resnet18(pretrained=False)
        self.download_weights_if_needed('./weights/resnet.pth', model_urls['resnet18'])
        self.gaze_network.load_state_dict(torch.load('./weights/resnet.pth'), strict=True)

        self.gaze_fc = torch.nn.Sequential(torch.nn.Linear(512, 2),)

    def forward(self, x):
        feature_list = self.gaze_network(x)
        out_feature = feature_list[-1]
        out_feature = out_feature.view(out_feature.size(0), -1)
        gaze = self.gaze_fc(out_feature)

        return gaze

    def download_weights_if_needed(self, weight_path, url):
        if not os.path.exists(weight_path):
            print(f"Downloading weights from {url} to {weight_path}")
            urllib.request.urlretrieve(url, weight_path)
