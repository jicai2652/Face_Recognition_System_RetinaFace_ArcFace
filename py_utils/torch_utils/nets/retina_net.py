import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride=1, leaky=0):
    """"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1x1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    """深度可分离卷积

    """
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    """Single Stage Headless捕获不同尺度的特征
    同时进行三组卷积:3*3, 5*5, 7*7
    """

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        # 四等分输出通道
        assert out_channel % 4 == 0
        # 根据输出通道数调整LeakyReLU斜率
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        # 3x3卷积分支, 没有ReLU激活
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        # 5x5卷积分支, 分解为两个连续的卷积, 第一个卷积使用ReLU激活
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        # 7x7卷积分支, 分解为两个连续的卷积, 第一个卷积使用ReLU激活
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input_channel):
        """前向传播
        执行三组卷积, 并合并结果
        """
        conv3X3 = self.conv3X3(input_channel)
        # 5x5, 第一个3x3卷积减少通道数, 第二个3*3卷积进行处理
        conv5X5_1 = self.conv5X5_1(input_channel)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        # 7x7, 第一个3x3卷积减少通道数, 第二个3*3卷积进行处理
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        # 合并结果, 并使用ReLU激活
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    """特征金字塔
    特征融合,对有效特征层调整通道数
    """
    def __init__(self, in_channels_list, out_channels):
        """初始化"""
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input_channel):
        # names = list(input_channel.keys())
        input_channel = list(input_channel.values())

        output1 = self.output1(input_channel[0])
        output2 = self.output2(input_channel[1])
        output3 = self.output3(input_channel[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        outputs = [output1, output2, output3]
        return outputs


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
