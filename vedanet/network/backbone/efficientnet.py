import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .. import layer as vn_layer
from .brick import darknet53 as bdkn

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

class SEModule(nn.Module):
    def __init__(self, in_chl, out_chl):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chl, out_chl, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(out_chl, in_chl, kernel_size=1, stride=1, padding=0, bias=True),
        )
    
    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

def conv_bn_act(in_, out_, kernel_size,
                stride=1, groups=1, bias=True,
                eps=1e-3, momentum=0.01):
    return nn.Sequential(
        SamePadConv2d(in_, out_, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm2d(out_, eps, momentum),
        Swish()
    )

class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv2d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, 1e-3, 0.01)
        )
        self.skip = skip and (stride == 1) and (in_ == out_)
        self.dropconnect = nn.Identity()
    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x

class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, modelname, 
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=1000):
        super().__init__()
        params = {
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        }

        width_coeff, depth_coeff , _, __ = params[modelname]
        print("You choose", modelname)

        min_depth = min_depth or depth_div

        self.stem = nn.Sequential(
            Conv2dSamePadding(3, 32, kernel_size=3,stride=2,bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.01, eps=0.001)
        )

        def renew_ch(x):
            if not width_coeff:
                return x

            new_x = x * width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * new_x:
                new_x += depth_div
            return new_x

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.blocks = nn.Sequential(
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        )
        self.fpn = nn.Sequential(
            bdkn.HeadBody(renew_ch(320), first_head=True),
            bdkn.Transition(160),
            bdkn.HeadBody(192),
            bdkn.Transition(64),
            bdkn.HeadBody(72),
        )

        # self.head = nn.Sequential(
        #     Conv2dSamePadding(renew_ch(320), renew_ch(1280), kernel_size=1, bias=False),
        #     nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
        #     nn.Linear(renew_ch(1280), num_classes)
        # )

        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, SamePadConv2d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out")
    #         elif isinstance(m, nn.Linear):
    #             init_range = 1.0 / math.sqrt(m.weight.shape[1])
    #             nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks[0](stem)
        stage2 = self.blocks[1](x)
        stage3 = self.blocks[2](stage2) # 40,52,52
        print(stage3.shape)
        x = self.blocks[3](stage3)
        stage4 = self.blocks[4](x)      # 112,26,26
        print(stage4.shape)
        x = self.blocks[5](stage4)
        stage5 = self.blocks[6](x)      # 320,13,13
        print(stage5.shape)

        head_body_1 =  self.fpn[0](stage5)                  # 160,13,13
        trans_1 = self.fpn[1](head_body_1)  # 80,26,26
        concat_2 = torch.cat([trans_1, stage4], 1)  # 192,26,26
        head_body_2 =  self.fpn[2](concat_2)                # 64,26,26
        trans_2 = self.fpn[3](head_body_2)  # 32,52,52
        concat_3 = torch.cat([trans_2, stage3], 1)  # 72,52,52
        head_body_3 =  self.fpn[4](concat_3)                # 24,52,52

        # stage 6, stage 5, stage 4
        features = [head_body_1, head_body_2, head_body_3]

        return features

if __name__ == "__main__":
    model = EfficientNet('efficientnet-b3')
    print(model)