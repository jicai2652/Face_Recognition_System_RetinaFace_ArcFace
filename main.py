#!/usr/bin/env python3
# coding: utf-8
"""
主程序,控制界面交互
"""
import os.path
import sys
# import time     # 识别计时功能(待实现)
# import logging       # 日志功能(待实现)
import cv2
import numpy as np
import torch
from PySide6.QtCore import QUrl, Slot, QObject, Signal
from PySide6.QtGui import QFontDatabase, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from py_utils import db_manager
from py_utils.initiate import ini_check, load_path
from py_utils.name_encode import name_encode
from py_utils.recog_progress import TorchRecogPipe#, OnnxRecogPipe
from py_utils.qt_utils import CvImageProvider, CameraWorker
import py_utils.qt_utils.compiled_resources

import faulthandler

# 启用faulthandler模块，以便在程序崩溃时打印堆栈跟踪
faulthandler.enable()


class Controller(QObject):
    """控制器"""

    def __init__(self, provider, cfg_path):
        super().__init__()
        self.cfg_path = cfg_path  # "resources/cfg.toml"
        # 信号发送渠道
        self.model_pipe = None
        self.records = None
        # self.face_db = None
        self.signal_bike = SignalBike()
        # 预览框图像显示
        self.provider = provider  # 图像提供设备
        self.provider.recogReady.connect(self.recog_operate)
        self.current_camera_mode = None  # 当前模式
        self.camera_worker = None
        self.is_camera_active = False
        # 运算设备设置
        self.device = None  # 初始为cpu,根据激活的版本进行切换
        # 数据库与识别
        self.db_path = os.path.abspath("database/face_data.db")
        self.current_recog_times = 0

    @Slot()
    def check_devices(self, torch_npu=None) -> None:
        framewk_cfg = load_path(self.cfg_path)  # 加载toml配置文件，输出一个字典
        print("当前启用的运算架构:\n\t", framewk_cfg["version"])
        if framewk_cfg["version"] == "pytorch":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            # elif torch_npu.npu.is_available():
            #     self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
            self.model_pipe = TorchRecogPipe(db_path=self.db_path, now_device=self.device, path_cfg=framewk_cfg)
        elif framewk_cfg["version"] == "ascend":
            self.device = "ascend"
            pass
        elif framewk_cfg["version"] == "onnx":
            self.device = "onnx"
            #self.model_pipe = OnnxRecogPipe(db_path=self.db_path, path_cfg=framewk_cfg)
        else:
            self.device = framewk_cfg["version"]
        self.signal_bike.set_now_device(str(self.device).upper())

    @Slot(str)
    def camera_trigger(self, send_type: str) -> None:
        try:
            # 切换模式时先释放旧资源
            if self.camera_worker:
                self.camera_release()
            # 创建新连接
            try:
                self.camera_worker = CameraWorker(0)
                self.camera_worker.start()
                # 建立新连接
                if send_type == "raw":
                    self.camera_worker.frame_ready.connect(self.provider.update_image)
                else:
                    self.camera_worker.frame_ready.connect(self.provider.update_recog_image)
                self.current_camera_mode = send_type
                self.is_camera_active = True
                self.signal_bike.cameraStatusChanged.emit(True)
            except Exception as e:
                print(f"摄像头初始化失败: {e}")
                self.camera_worker = None
        except Exception as e:
            print(f"摄像头初始化失败: {e}")

    @Slot()
    def camera_release(self) -> None:
        if not self.camera_worker:
            return
        self.camera_worker.frame_ready.disconnect()
        self.camera_worker.stop()
        if self.camera_worker.isRunning():
            self.camera_worker.wait(300)
        if self.camera_worker.isFinished():
            self.camera_worker.deleteLater()
        # 重置状态
        self.camera_worker = None
        self.is_camera_active = False
        self.signal_bike.cameraStatusChanged.emit(False)
        self.current_camera_mode = None

    @Slot()
    def recog_start(self) -> None:
        """触发信号时，开始识别：为了避免直接识别内存越界，特此封装"""
        try:
            if self.recog_operate(self.provider.current_frame):
                pass
        except Exception as ce:
            print(f"问题{ce}")
            return

    def recog_operate(self, frame: np.ndarray) -> bool:
        try:

            finds = self.model_pipe.recog_operate(frame)
            for (find_info, find_path) in finds:
                self.provider.update_recog_view(find_info, find_path)
                # 统计记录
                self.current_recog_times += 1
                self.update_today_recog()
                self.update_total_recog()
            return True
        except  Exception as e:
            print(f"识别过程异常: {e}")
            return False

    @Slot(str, str)
    def take_picture(self, user_id, user_name) -> None:
        print("信息录入：",user_id, user_name)
        utf_name = name_encode(user_name)
        save_path = "database/face_pic/" + user_id + "_" + utf_name + ".jpg"
        frame = self.provider.current_frame

        with torch.no_grad():
            face_pics = self.model_pipe.face_align_muti(frame)  # .face_align(frame)  # 此时会自动转换为BGR格式
            # 操作np数组实现通道转换, face_pic输出为RGB格式的裁剪后图片,需要转换为BGR以便openCV保存
            save_pic = np.array(face_pics[0])[..., ::-1]
            feature = self.model_pipe.feature_extract(face_pics[0])
            db_manager.add_msg_to_db("database/face_data.db", user_id, user_name, feature, save_path)
            cv2.imwrite(str(save_path), save_pic)

    @Slot()
    def load_table_db(self) -> None:
        """加载数据库,在QML表格视图中展示编号,姓名,识别次数"""
        self.records = db_manager.load_records(self.db_path)
        if self.records:
            self.signal_bike.set_db_table(self.records)

    @Slot(str, str)
    def delete_user(self, uid, name) -> None:
        """删除用户"""
        if db_manager.delete_msg_from_db(self.db_path, uid, name):
            print(f"用户 {uid} ({name}) 已被删除")
        else:
            print(f"未能成功删除用户 {uid} ({name}) ")

    @Slot()
    def update_total_users(self):
        """连接数据库读取总用户数"""
        total_users = db_manager.get_total_users(self.db_path)
        self.signal_bike.set_total_users(str(total_users))

    @Slot()
    def update_total_recog(self) -> None:
        total_recog = db_manager.get_total_recog(self.db_path)
        self.signal_bike.set_total_recog(str(total_recog))

    @Slot()
    def update_today_recog(self) -> None:
        self.signal_bike.set_today_recog(str(self.current_recog_times))

    @Slot()
    def update_device(self) -> None:
        self.signal_bike.set_now_device(str(self.device).upper())

    @Slot()
    def update_documents(self) -> None:
        self.signal_bike.set_file_path("config.toml", "readme.md")


class SignalBike(QObject):
    """信号发送到界面"""
    # 首页信息卡片
    totalUsersChanged = Signal(str)
    todayRecogChanged = Signal(str)
    totalRecogChanged = Signal(str)
    nowDeviceChanged = Signal(str)
    filePathChanged = Signal(str, str)
    tableDBChanged = Signal('QVariantList')  # 发送字典到QML处理:分拆成列表后在表格视图显示
    # Controller 直发
    userLoaded = Signal(str, str)  # 用户加载
    cameraStatusChanged = Signal(bool)

    def set_total_users(self, total_users: str) -> None:
        self.totalUsersChanged.emit(total_users)

    def set_today_recog(self, today_recog: str) -> None:
        self.todayRecogChanged.emit(today_recog)

    def set_total_recog(self, total_recog: str) -> None:
        self.totalRecogChanged.emit(total_recog)

    def set_now_device(self, now_device: str) -> None:
        self.nowDeviceChanged.emit(now_device)

    def set_file_path(self, config_path: str, guide_path: str) -> None:
        self.filePathChanged.emit(config_path, guide_path)

    def set_db_table(self, records: list[dict]) -> None:
        self.tableDBChanged.emit(records)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(':/icons/logo.png'))
    font_id = QFontDatabase.addApplicationFont(":/fonts/HarmonyOS_SansSC_Medium.ttf")
    if font_id == -1:
        print("Failed to load font")
        sys.exit(1)
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
    app.setFont(font_family)
    engine = QQmlApplicationEngine()
    try:
        # 初始化
        cfg_path = os.path.abspath("resources/cfg.toml")
        ini_check(cfg_path)
        cv_provider = CvImageProvider()
        controller = Controller(cv_provider, cfg_path)
        engine.addImageProvider("cvProvider", cv_provider)
        engine.rootContext().setContextProperty("cvProvider", cv_provider)
        engine.rootContext().setContextProperty("controller", controller)
        engine.rootContext().setContextProperty("signalBike", controller.signal_bike)
        # 加载QML
        engine.load(QUrl.fromLocalFile("resources/main_window.qml"))
        if not engine.rootObjects():
            print("Failed to load QML file")
            return -1
        # 连接信号
        main_window = engine.rootObjects()[0]
        main_window.image_path.connect(cv_provider.update_image_from_path)

        sys.exit(app.exec())
    except Exception as syse:
        print(f"发生异常:{syse}")
    finally:
        print("bye")

    return None


if __name__ == "__main__":
    main()
