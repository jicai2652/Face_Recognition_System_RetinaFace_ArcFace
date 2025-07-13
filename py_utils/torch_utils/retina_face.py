#!/usr/bin/env python3
# coding: utf-8
"""
RetinaFace
参考自：https://github.com/biubug6/Pytorch_Retinaface/blob/master/models/retinaface.py
增改内容：检测框
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils

from py_utils.cv_utils.py_cpu_nms import py_cpu_nms
from py_utils.cv_utils.align_trans import get_reference_facial_points, warp_and_crop_face
from py_utils.torch_utils.utilsRetina.prior_box import PriorBox
from py_utils.torch_utils.nets.retina_net import FPN, SSH, MobileNetV1
from py_utils.torch_utils.utilsRetina.box_utils import decode, decode_landm  # , nms


class ClassHead(nn.Module):
    """分类
    确认先验框中有没有人脸__使用SoftMax分类,保留两个分数以表示类别概率
    """

    def __init__(self, inchannels=512, num_anchors=3):
        """初始化"""
        super(ClassHead, self).__init__()
        # self.num_anchors = num_anchors
        # 每个锚点用两个分数表示类别概率,输出通道数调整为2
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        """前向传播"""
        out = self.conv1x1(x)
        # 调整输出的维度顺序
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    """锚框
    调整先验框参数,获取bounding box
    """

    def __init__(self, inchannels=512, num_anchors=3):
        """初始化"""
        super(BboxHead, self).__init__()
        # 每个锚框需要4个坐标值, 通道数调整为4
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        """前向传播"""
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    """关键点"""

    def __init__(self, inchannels=512, num_anchors=3):
        """初始化"""
        super(LandmarkHead, self).__init__()
        # 五个关键点共十个坐标
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        """前向传播"""
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='test', device=None):
        """初始化
        :param cfg:  模型预设
        :param phase: 训练/测试
        :param device: 外部指定torch设备
        """
        super(RetinaFace, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.resize = 1
        self.vis_threshold = 0.6
        self.nms_threshold = 0.4
        self.confidence_threshold = 0.5
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda") if device is None else device
        backbone = None
        if self.cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if self.cfg['pretrain']:
                checkpoint = torch.load(self.cfg['pretrained_path'], map_location=self.device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        self.body = _utils.IntermediateLayerGetter(backbone, self.cfg['return_layers'])
        in_channels_stage2 = self.cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = self.cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=self.cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=self.cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=self.cfg['out_channel'])

    @staticmethod
    def _make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
        """FPN分类"""
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    @staticmethod
    def _make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
        """FPN锚框"""
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    @staticmethod
    def _make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
        """FPN关键点"""
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        """前向传播
        计算预测框->是否包含人脸->获取BoundingBox与关键点
        :param inputs:tensor(batch_size, channels, height, width)
        :return:
            tuple(预测框，分类概率，关键点)
        """
        out = self.body(inputs)
        # FPN
        fpn = self.fpn(out)
        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        # 对先验框并行计算：bbox回归, 类别概率, 关键点回归
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    def get_dets(self, frame):
        """获取检测框 BoundingBox 与关键点

        :param frame: 图像nparray(H,W,C)
        :return: 点集与置信度, 关键点坐标
        """
        img = np.float32(frame)
        # 处理图像尺度,便于预测框对齐
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)  # 减去均值
        img = img.transpose(2, 0, 1)  # 转换为 CHW 格式
        img = torch.from_numpy(img).unsqueeze(0)  # 添加 batch 维度
        # 加载数据
        img = img.to(self.device)
        scale = scale.to(self.device)
        # 前向传播
        loc, conf, landmks = self.forward(img)
        # 生成先验框
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        # 先验框解码，预测边界框
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        # 每个bbox的置信度分数
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # 关键点
        landmks = decode_landm(landmks.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landmks = landmks * scale1 / self.resize
        landmks = landmks.cpu().numpy()
        # 忽略低置信度的预测结果
        indices = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[indices]
        landmks = landmks[indices]
        scores = scores[indices]
        # 结果预测框(可留k个最佳项)
        order = scores.argsort()[::-1]  # [:args.top_k]
        boxes = boxes[order]
        landmks = landmks[order]
        scores = scores[order]
        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landmks = landmks[keep]
        return dets, landmks


    # def face_align_muti(self, frame):
    #     """多人脸对齐
    #     接收一帧图像作为输入,使用模型检测多个人脸并对齐,返回对齐后的多个人脸图像列表
    #     :param frame: 输入的图像帧
    #     :return: 对齐后的人脸图像列表,如果没有检测到人脸则返回空列表
    #     """
    #     dets, landmks = self.get_dets(frame)
    #     if len(dets) == 0:
    #         return []
    #
    #     warped_faces = []
    #     reference = get_reference_facial_points(default_square=True)
    #
    #     for i in range(len(dets)):
    #         # 提取第i个人脸的关键点
    #         facial5points = []
    #         ij = 0
    #         for j in range(5):
    #             l1 = [landmks[i][ij], landmks[i][ij + 1]]
    #             facial5points.append(l1)
    #             ij += 2
    #
    #         # 对齐人脸
    #         warped_face = warp_and_crop_face(frame, facial5points, reference, crop_size=(112, 112))
    #         warped_faces.append(warped_face)
    #
    #     return warped_faces
    #
