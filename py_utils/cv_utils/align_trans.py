"""
人脸对齐和变换工具 (带详尽注释版本)
参考自: https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/mtcnn_pytorch/src/align_trans.py

本模块提供人脸对齐和变换功能，主要用于将检测到的人脸关键点对齐到标准位置。
包含以下主要功能：
1. 获取标准人脸关键点坐标
2. 计算仿射变换矩阵
3. 应用变换并裁剪人脸区域
"""

import cv2
import numpy as np
from py_utils.torch_utils.utilsRetina.matlab_cp2tform import get_similarity_transform_for_cv2

# 标准人脸5点坐标 (x,y)，用于对齐的参考点
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],  # 左眼中心
    [65.53179932, 51.50139999],  # 右眼中心
    [48.02519989, 71.73660278],  # 鼻尖
    [33.54930115, 92.3655014],  # 左嘴角
    [62.72990036, 92.20410156]  # 右嘴角
]
DEFAULT_CROP_SIZE = (96, 112)  # 默认裁剪尺寸 (宽,高)


def get_reference_facial_points(output_size=None, inner_padding_factor=0.0,
                                outer_padding=(0, 0), default_square=False) -> np.ndarray:
    """根据裁剪设置获取标准的5个关键点坐标"""
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)
    # 步骤1: 如果需要正方形区域，调整关键点位置
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if output_size and output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]:
        # print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    # 如果没有填充且输出尺寸为默认值，直接返回
    if inner_padding_factor == 0 and outer_padding == (0, 0):
        if output_size is None:
            return tmp_5pts
        else:
            raise Exception('无填充时output_size必须为None或{}'.format(tmp_crop_size))
    # 检查参数有效性
    if not (0 <= inner_padding_factor <= 1.0):
        raise Exception('inner_padding_factor必须在[0,1]范围内')
    # 步骤2: 应用内部填充
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)
    # 步骤3: 计算缩放因子并调整关键点
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    # 步骤4: 应用外部填充
    reference_5point = tmp_5pts + np.array(outer_padding)
    return reference_5point

def get_affine_transform_matrix(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """计算从源点到目标点的仿射变换矩阵"""
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])  # 默认单位矩阵
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])  # 添加齐次坐标
    dst_pts_ = np.hstack([dst_pts, ones])
    # 最小二乘法求解
    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_, rcond=None)
    # 根据矩阵秩构造变换矩阵
    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])
    return tfm


def warp_and_crop_face(src_img: np.ndarray, facial_pts: list, reference_pts: list, crop_size: tuple = (96, 112),
                       align_type: str = 'affine') -> np.ndarray:
    # pass
    """对人脸区域进行变换和裁剪"""
    # 获取参考关键点
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            reference_pts = get_reference_facial_points(
                output_size=crop_size,
                inner_padding_factor=0,
                outer_padding=(0, 0),
                default_square=False)
    # 转换点坐标
    ref_pts = np.float32(reference_pts)
    if ref_pts.shape[0] == 2:  # 如果是(2,K)格式转置为(K,2)
        ref_pts = ref_pts.T
    src_pts = np.float32(facial_pts)
    if src_pts.shape[0] == 2:  # 如果是(2,K)格式转置为(K,2)
        src_pts = src_pts.T
    # 检查点坐标形状
    if src_pts.shape != ref_pts.shape:
        raise Exception('人脸关键点和参考关键点形状不一致')
    # 根据变换类型计算变换矩阵
    if align_type == 'perspective':
        # 透视变换需要4个点
        if len(src_pts) == 5:
            src_pts = [src_pts[0], src_pts[1], src_pts[3], src_pts[4]]
            ref_pts = [ref_pts[0], ref_pts[1], ref_pts[3], ref_pts[4]]
        tfm = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(ref_pts))
        face_img = cv2.warpPerspective(src_img, tfm, (crop_size[0], crop_size[1]))
    else:
        # 计算仿射变换矩阵
        if align_type == 'cv2_affine':
            tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
        elif align_type == 'affine':
            tfm = get_affine_transform_matrix(src_pts, ref_pts)
        else:
            tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
        # 应用仿射变换
        face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img
