import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from py_utils.torch_utils.nets.iresnet import iresnet50


# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ArcMarginProduct, self).__init__()
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin  # 角度间隔
        self.scale = scale  # 缩放尺度
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # 通过平均分布初始化权重，以图避免梯度问题
        # xavier初始化->根据输入/输出神经元的数量调整权重的初始范围,控制传播时的方差

    #       # xavier_uniform: 按照均匀分布初始化
    def forward(self, inputs, labels):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))

        arcface_loss = F.cross_entropy(self.scale * (cosine - self.margin * labels), labels)

        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cosine * self.cos_m - sine * self.sin_m
        # phi = torch.where(cosine > self.cos_m, phi, cosine - self.mm)
        # one_hot = torch.zer
        # cosine = F.linear(inputs, F.normalize(self.weight))
        # # .clamp: 将张量的值限定在输入参数范围内(将sin^2的值限定在0~1之间)
        # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # # cos(θ+m) = cos(θ)cos(m) - sin(θ)sin(m)
        # phi = cosine * self.cos_m - sine * self.sin_m
        # # torch.where 根据条件选择元素
        # phi = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)
        # # one_hot标签阵
        # one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output *= self.s
        return arcface_loss


# class ArcfaceHead(nn.Module):
#     """特征归一化、角度边缘损失函数"""
#
#     def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
#         """初始化
#
#         :param embedding_size: 特征向量维度(128)
#         :param s: 缩放因子(64)控制输出结果的尺度
#         :param m: 角度边缘值（分类间隔）默认0.5rad -->cos(θ+m)式中的m
#         """
#         super(ArcfaceHead, self).__init__()
#         self.s = s
#         self.m = m
#         # 初始化权重,xavier初始化->根据输入/输出神经元的数量调整权重的初始范围,控制传播时的方差
#         # xavier_uniform:按照均匀分布初始化
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
#         nn.init.xavier_uniform_(self.weight)
#
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#
#     def forward(self, inputs, label):
#         """前向传播"""
#         # F.linear相当于矩阵乘法,计算输入参数的点积
#         cosine = F.linear(inputs, F.normalize(self.weight))
#         # .clamp: 将张量的值限定在输入参数范围内(将sin^2的值限定在0~1之间)
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         # cos(θ+m) = cos(θ)cos(m) - sin(θ)sin(m)
#         phi = cosine * self.cos_m - sine * self.sin_m
#         # torch.where 根据条件选择元素
#         phi = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)
#         # one_hot标签阵
#         one_hot = torch.zeros(cosine.size()).type_as(phi).long()
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
#         return output


class Arcface(nn.Module):
    def __init__(self, backbone="iResNet50", pretrained=False):
        super(Arcface, self).__init__()
        if backbone == "iResNet50":
            self.embedding_size = 512
            self.s = 64
            self.arcface = iresnet50(dropout_keep_prob=0.5, embedding_size=self.embedding_size, pretrained=pretrained)

    def forward(self, x, y=None, mode="predict"):
        """前向传播"""
        # 通过模型提取特征
        x = self.arcface(x)
        # .view:改变张量的形状(这里展平到2维) 第一个参数为形状, 第二个参数表示自动推到剩余的维度(这里是是展平到一维)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        return x
