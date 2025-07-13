#!/usr/bin/env python3
# coding: utf-8
"""
取自：https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/nms/py_cpu_nms.py
"""
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np


def py_cpu_nms(dets:np.ndarray, thresh:float)->list:
    """Python实现的NMS"""
    # 切片
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    conf_scores = dets[:, 4]
    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 置信度从大到小排序
    order = conf_scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        # 保留最高分
        i = order[0]
        keep.append(i)
        # 计算当前框与其他框体的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算交并比（IOU)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IOU小于阈值的检测框
        indices = np.where(ovr <= thresh)[0]
        order = order[indices + 1]

    return keep
