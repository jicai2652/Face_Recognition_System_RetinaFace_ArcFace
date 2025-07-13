import math
import os.path

import cv2
# import onnxruntime
import torch
import numpy as np

from py_utils import db_manager
from py_utils.cv_utils.align_trans import warp_and_crop_face, get_reference_facial_points
from py_utils.cv_utils.det_convert import decode, decode_landmks
from py_utils.torch_utils.arcface import ArcFace
from py_utils.torch_utils.retina_face import RetinaFace
from py_utils.torch_utils.face_match import detect_db_batch_muti, detect_db_batch_no_torch
from py_utils.initiate import cfg_mnet
from py_utils.torch_utils.utilsRetina.model_call import load_model

# from py_utils.ascend_utils.acl_load import init_acl, deinit_acl
# from py_utils.cv_utils.det_convert import letterbox_image  # , filter_box
from py_utils.cv_utils.py_cpu_nms import py_cpu_nms


class TorchRecogPipe:
    def __init__(self, now_device: torch.device, db_path: str, det_cfg=cfg_mnet, path_cfg=None):
        self.device = torch.device(now_device)
        self.db_path = db_path
        self.records = []
        self.det_model = None
        self.det_cfg = det_cfg
        self.det_model_path = path_cfg["retina"]  # "resources/models/mobilenet0.25_Final.pth"
        self.rec_model = None
        self.rec_model_path = path_cfg["arcface"]  # "resources/models/acc_9737.pth"

        self.model_init(self.det_cfg)

    def model_init(self, det_cfg):
        self.det_model = RetinaFace(det_cfg, phase='test', device=self.device)
        self.det_model = load_model(self.det_model, self.det_model_path, self.device)
        self.det_model.eval()
        self.det_model = self.det_model.to(self.device)
        self.rec_model = ArcFace(self.rec_model_path, device=self.device)

    def face_align_muti(self, frame):
        """多人脸对齐
        接收一帧图像作为输入,使用模型检测多个人脸并对齐,返回对齐后的多个人脸图像列表
        :param frame: 输入的图像帧
        :return: 对齐后的人脸图像列表,如果没有检测到人脸则返回空列表
        """
        dets, landmks = self.det_model.get_dets(frame)
        if len(dets) == 0:
            return []
        warped_faces = []
        reference = get_reference_facial_points(default_square=True)
        for i in range(len(dets)):
            # 提取第i个人脸的关键点
            facial5points = []
            ij = 0
            for j in range(5):
                l1 = [landmks[i][ij], landmks[i][ij + 1]]
                facial5points.append(l1)
                ij += 2
            # 对齐人脸
            warped_face = warp_and_crop_face(frame, facial5points, reference, crop_size=(112, 112))
            warped_faces.append(warped_face)
        return warped_faces

    def feature_extract(self, face_area: np.ndarray) -> np.ndarray:
        return self.rec_model.feature_extract(face_area)

    def recog_operate(self, frame: np.ndarray) -> list[tuple[str, str]]:
        try:
            with torch.no_grad():
                face_img_list = self.face_align_muti(frame)
                face_img_list = self.det_model.face_align_muti(frame)
                if not face_img_list:
                    print("未检测到人脸")
                    return []
                print(f"检测到{len(face_img_list)}张人脸，开始识别")
                features = [self.rec_model.feature_extract(face_img) for face_img in face_img_list]
                results = detect_db_batch_muti(self.db_path, features, self.device)
                print(f"数据库匹配返回值: {results}")
                output = []
                for (find_id, find_name, similarity, find_path) in results:
                    if similarity == 0.0:
                        print(f"未找到匹配用户: {find_id}")
                        continue
                    record_time = ''  # 记录时间，未完成
                    if db_manager.add_record_to_db(self.db_path, find_id, find_name, record_time):
                        find_info = f"{find_id}_{find_name}"
                        print(f"识别记录已录入：{find_info}")
                        output.append((find_info, find_path))
                return output
        except Exception as e:
            print(f"识别过程匹配异常: {e}")
            return []

    def release_models(self):
        """释放模型资源"""
        if self.det_model is not None:
            del self.det_model
            self.det_model = None
        if self.rec_model is not None:
            del self.rec_model
            self.rec_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("已释放模型")


# class OnnxRecogPipe:
#     def __init__(self, db_path: str,path_cfg=None):
#         self.db_path = db_path
#         self.det_model = None
#         self.rec_model = None
#         self.det_model_path = path_cfg["retina"]    # './resources/models/mobilenet0.25_Final.onnx'
#         self.rec_model_path = path_cfg["arcface"]   # './resources/models/acc_9737.onnx'
#         self.model_init()
#
#     def model_init(self):
#         self.det_model = onnxruntime.InferenceSession(self.det_model_path)
#         self.rec_model = onnxruntime.InferenceSession(self.rec_model_path)
#         print('初始化模型成功')
#
#     def get_input_name(self, now_model):
#         return [node.name for node in now_model.get_inputs()]
#
#     def get_output_name(self):
#         return [node.name for node in self.det_model.get_outputs()]
#
#     def get_input_feed(self, now_model, img_tensor):
#         input_feed = {}
#         for name in self.get_input_name(now_model):
#             input_feed[name] = img_tensor
#         return input_feed
#
#     def generate_anchors(self, image_size):
#         # 取整得点
#         feature_maps = [[math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)] for step in
#                         cfg_mnet['steps']]
#         anchors = []
#         for k, f in enumerate(feature_maps):
#             min_sizes = cfg_mnet['min_sizes'][k]
#             for i in range(f[0]):
#                 for j in range(f[1]):
#                     for min_size in min_sizes:
#                         s_kx = min_size / image_size[1]
#                         s_ky = min_size / image_size[0]
#                         dense_cx = [x * cfg_mnet['steps'][k] / image_size[1] for x in [j + 0.5]]
#                         dense_cy = [y * cfg_mnet['steps'][k] / image_size[0] for y in [i + 0.5]]
#                         for cy in dense_cy:
#                             for cx in dense_cx:
#                                 anchors += [cx, cy, s_kx, s_ky]
#         return np.array(anchors).reshape(-1, 4)
#
#     def get_dets(self, frame: np.ndarray):
#         # 预处理
#         img = letterbox_image(frame, (640, 640))
#         or_img = np.array(img, np.uint8)
#
#         img = img.astype(dtype=np.float32)
#         img -= np.array((104, 117, 123), np.float32)
#         img = img[:, :, ::-1]  # BGR转RGB
#         img = img.transpose(2, 0, 1)  # HWC转CHW
#         img = np.expand_dims(img, axis=0)
#
#         # 推理
#         input_feed = self.get_input_feed(self.det_model, img)
#         pred = self.det_model.run(None, input_feed)
#         # 后处理
#         output_1 = np.array(pred[0]).squeeze()
#         output_2 = np.array(pred[1]).squeeze()
#         output_3 = np.array(pred[2]).squeeze()
#         anchors = self.generate_anchors((640, 640))
#         boxes = decode(output_1, anchors, cfg_mnet['variance'])
#         landmks = decode_landmks(output_3, anchors, cfg_mnet['variance'])
#         conf = output_2[:, 1:2]
#         boxs_conf = np.concatenate((boxes, conf, landmks), -1)
#         boxs_conf = filter_box(boxs_conf, 0.5, 0.5)
#         # 缩放回原图尺寸
#         boxs_conf[:, :4] = boxs_conf[:, :4] * 640
#         boxs_conf[:, 5:] = boxs_conf[:, 5:] * 640
#
#         return boxs_conf, or_img
#
#     def face_align_muti(self, frame: np.ndarray) -> list[np.ndarray]:
#         """多人脸对齐
#         接收一帧图像作为输入，使用ONNX模型检测多个人脸并对齐，返回对齐后的多个人脸图像列表
#         :param frame: 输入的图像帧
#         :return: 对齐后的人脸图像列表,如果没有检测到人脸则返回空列表
#         """
#         try:
#             # 获取检测结果并验证格式
#             boxs_conf, or_img = self.get_dets(frame)
#             if not isinstance(boxs_conf, np.ndarray) or boxs_conf.size == 0:
#                 print("ONNX模型未检测到人脸或输出格式无效")
#                 return []
#             print(f"ONNX模型检测到{len(boxs_conf)}张人脸，开始对齐")
#             warped_faces = []
#             reference = get_reference_facial_points(default_square=True)
#             for i, box in enumerate(boxs_conf):
#                 try:
#                     # 获取5个人脸关键点 [左眼,右眼,鼻尖,左嘴角,右嘴角]
#                     landmarks = box[5:15].reshape(5, 2).astype(np.float32)
#
#                     # 使用与Torch版本相同的对齐参数
#                     warped_face = warp_and_crop_face(
#                         or_img,
#                         landmarks,
#                         reference,
#                         crop_size=(112, 112)
#                     )
#                     warped_faces.append(warped_face)
#                 except Exception as e:
#                     print(f"处理第{i}个人脸时出错: {str(e)}")
#                     continue
#
#             if not warped_faces:
#                 print("ONNX模型未能成功对齐任何人脸")
#             return warped_faces
#
#         except Exception as e:
#             print(f"ONNX多人脸对齐异常: {str(e)}")
#             return []
#
#     def recog_operate(self, frame: np.ndarray) -> list[tuple[str, str]]:
#         """识别执行"""
#         try:
#             # 人脸检测与对齐
#             warped_faces = self.face_align_muti(frame)
#             # 特征提取
#             features = [self.feature_extract(face_img) for face_img in warped_faces]
#             # 数据库比对
#             results = detect_db_batch_no_torch(self.db_path, features)
#             print(f"识别到用户: {results}")
#             output = []
#             for (find_id, find_name, similarity, find_path) in results:
#                 if similarity == 0.0:
#                     print(f"未找到匹配用户: {find_id}")
#                     continue
#                 record_time = ''  # 记录时间，未完成
#                 if db_manager.add_record_to_db(self.db_path, find_id, find_name, record_time):
#                     find_info = f"{find_id}_{find_name}"
#                     print(f"{find_info}识别记录已录入")
#                     output.append((find_info, find_path))
#             return output
#         except Exception as e:
#             print(f"ONNX识别过程异常: {e}")
#             return []
#
#     def feature_extract(self, aligned_face: np.ndarray) -> np.ndarray | None:
#         """ONNX特征提取"""
#         try:
#             # 预处理
#             image = np.array(aligned_face).astype(np.float32)
#             image = (image - 127.5) / 128.0  # 归一化到[-1, 1]
#             image = image.transpose(2, 0, 1)  # HWC -> CHW
#             image = np.expand_dims(image, axis=0)  # 添加batch维度
#             # 获取输入名称
#             input_name = self.rec_model.get_inputs()[0].name
#
#             # 特征提取
#             feature = self.rec_model.run(None, {input_name: image})[0]
#
#             # 验证特征维度
#             if feature.shape[1] != 512:
#                 raise ValueError(f"特征维度错误，应为512维，实际得到{feature.shape[1]}维")
#
#             return feature
#         except Exception as e:
#             print(f"ONNX特征提取失败：{e}")
#             return None

def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    conf = org_box[..., 4] > conf_thres  # 删除置信度小于conf_thres的BOX
    box = org_box[conf == True]
    output = []
    curr_cls_box = np.array(box)
    curr_out_box = py_cpu_nms(curr_cls_box, iou_thres)  # 经过非极大抑制后输出的BOX下标
    for k in curr_out_box:
        output.append(curr_cls_box[k])  # 利用下标取出非极大抑制后的BOX
    output = np.array(output)
    return output
