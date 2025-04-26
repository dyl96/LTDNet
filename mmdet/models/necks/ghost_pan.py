# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

"""
2020.06.09-Changed for building GhostNet
Huawei Technologies Co., Ltd. <foss@huawei.com>
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang,
Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
and https://github.com/rwightman/pytorch-image-models
"""
import logging
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

import torch.nn as nn

import torch.nn as nn

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "SyncBN": ("bn", nn.SyncBatchNorm),
    "GN": ("gn", nn.GroupNorm),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)

activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "SiLU": nn.SiLU,
    "HardSwish": nn.Hardswish,
    "Hardswish": nn.Hardswish,
    None: nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        return activations[name](inplace=True)


def get_url(width_mult=1.0):
    if width_mult == 1.0:
        return "https://raw.githubusercontent.com/huawei-noah/CV-Backbones/master/ghostnet_pytorch/models/state_dict_73.98.pth"  # noqa E501
    else:
        logging.info("GhostNet only has 1.0 pretrain model. ")
        return None


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        activation="ReLU",
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layers(activation)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, activation="ReLU"):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layers(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, activation="ReLU"
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            act_layers(activation) if activation else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            act_layers(activation) if activation else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        activation="ReLU",
        se_ratio=0.0,
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, activation=activation)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, activation=None)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(
        self,
        width_mult=1.0,
        out_stages=(4, 6, 9),
        activation="ReLU",
        pretrain=True,
        act=None,
    ):
        super(GhostNet, self).__init__()
        assert set(out_stages).issubset(i for i in range(10))
        self.width_mult = width_mult
        self.out_stages = out_stages
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t,   c,  SE, s
            # stage1
            [[3, 16, 16, 0, 1]],  # 0
            # stage2
            [[3, 48, 24, 0, 2]],  # 1
            [[3, 72, 24, 0, 1]],  # 2  1/4
            # stage3
            [[5, 72, 40, 0.25, 2]],  # 3
            [[5, 120, 40, 0.25, 1]],  # 4  1/8
            # stage4
            [[3, 240, 80, 0, 2]],  # 5
            [
                [3, 200, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
            ],  # 6  1/16
            # stage5
            [[5, 672, 160, 0.25, 2]],  # 7
            [
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
            ],  # 8
        ]
        #  ------conv+bn+act----------# 9  1/32

        self.activation = activation
        if act is not None:
            warnings.warn(
                "Warning! act argument has been deprecated, " "use activation instead!"
            )
            self.activation = act

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = act_layers(self.activation)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        activation=self.activation,
                        se_ratio=se_ratio,
                    )
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width_mult, 4)
        stages.append(
            nn.Sequential(
                ConvBnAct(input_channel, output_channel, 1, activation=self.activation)
            )
        )  # 9

        self.blocks = nn.Sequential(*stages)

        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        output = []
        for i in range(10):
            x = self.blocks[i](x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "conv_stem" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if pretrain:
            url = get_url(self.width_mult)
            if url is not None:
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict, strict=False)


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str): activation layer, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        activation="ReLU",
        inplace=True,
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = nn.Conv2d(  #
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and self.activation:
                x = self.act(x)
        return x


class DepthwiseConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias="auto",
        norm_cfg=dict(type="BN"),
        activation="ReLU",
        inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(DepthwiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == {
            "depthwise",
            "dwnorm",
            "act",
            "pointwise",
            "pwnorm",
            "act",
        }

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.depthwise, nonlinearity=nonlinearity)
        kaiming_init(self.pointwise, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer_name in self.order:
            if layer_name != "act":
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == "act" and self.activation:
                x = self.act(x)
        return x


class GhostBlocks(nn.Module):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expand=1,
        kernel_size=5,
        num_blocks=1,
        use_res=False,
        activation="LeakyReLU",
    ):
        super(GhostBlocks, self).__init__()
        self.use_res = use_res
        if use_res:
            self.reduce_conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=activation,
            )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels,
                    int(out_channels * expand),
                    out_channels,
                    dw_kernel_size=kernel_size,
                    activation=activation,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out

from mmcv.runner import BaseModule
from ..builder import NECKS

@NECKS.register_module()
class GhostPAN(BaseModule):
    """Path Aggregation Network with Ghost block.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        num_extra_level (int): Number of extra conv layers for more feature levels.
            Default: 0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        activation (str): Activation layer name.
            Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_depthwise=False,
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        norm_cfg=dict(type="BN"),
        activation="LeakyReLU",
    ):
        super(GhostPAN, self).__init__()
        assert num_extra_level >= 0
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    out_channels,
                    1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    activation=activation,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.bottom_up_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    activation=activation,
                )
            )

        # extra layers
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        for i in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.extra_lvl_out_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: multi level features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = [
            reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)
        ]
        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        # extra layers
        for extra_in_layer, extra_out_layer in zip(
            self.extra_lvl_in_conv, self.extra_lvl_out_conv
        ):
            outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))

        return tuple(outs)