#! /usr/bin/env python3
# coding: utf-8
"""
增改内容：多平台支持与特征抽取
参考自: https://github.com/bubbliiiing/arcface-pytorch/blob/main/arcface.py
"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from py_utils.torch_utils.nets.arc_net import Arcface as arcFace


def preprocess_input(image):
    image /= 255.0  # 颜色归一到[0,1]
    image -= 0.5
    image /= 0.5
    return image  # 色彩通道[-1, 1]


class ArcFace(object):

    def __init__(self, model_path, device):
        self.net = None
        self.input_shape = [112, 112, 3]
        self.device = torch.device(device)  # 确保device是一个torch.device对象
        self.backbone = "iResNet50"
        self.letterbox_image = True
        self.model_path = model_path
        self.generate()

    def generate(self):
        self.net = arcFace(backbone=self.backbone).eval()
        # 确保模型权重加载到指定设备
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(state_dict, strict=False)
        self.net = self.net.to(self.device)  # 显式将模型移动到指定设备
        print(f"ArcFace模型已加载到设备{self.device}:\t{self.model_path}")
        if self.device.type == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

    def feature_extract(self, image:np.ndarray)->np.ndarray:
        photo = torch.from_numpy(
            np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0))
        # 确保输入数据与模型在同一设备上
        if photo.device != self.device:
            photo = photo.to(self.device)
        with torch.no_grad():
            output = self.net(photo).cpu().numpy()
        return output