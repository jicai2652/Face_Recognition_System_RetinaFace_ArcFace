# image_provider.py
# import os
import os
import time
import cv2
import numpy as np
from PySide6.QtCore import QUrl, Signal, QThread, Slot, QMutex
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider
import logging
# from py_utils.recog_progress import TorchRecogPipe
import faulthandler

faulthandler.enable()


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)  # 用以发送帧图像给 CvImageProvider

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        self._running = False
        self.mutex = QMutex()

    @Slot()
    def run(self):
        try:
            self.mutex.lock()
            if self.cap is None:
                # logging.log("摄像头初始化中...")
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_ANY)
            self.mutex.unlock()

            if not self.cap.isOpened():
                print(f"摄像头{self.camera_id}寄了，大侠请重新来过!")
                return

            self._running = True
            fps_timer = time.time()
            frame_cnt = 0

            while self._running:
                self.mutex.lock()
                if not self._running:
                    self.mutex.unlock()
                    break
                self.mutex.unlock()

                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("摄像头读取失败")
                    QThread.msleep(10)
                    continue
                try:
                    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_flp = cv2.flip(frame, 1)
                    self.mutex.lock()
                    frame_cpy = frame_flp.copy()
                    self.mutex.unlock()
                    self.frame_ready.emit(frame_cpy)

                    frame_cnt += 1
                    if frame_cnt >= 30:
                        now = time.time()
                        elapsed = now - fps_timer
                        fps_timer = now
                        frame_cnt = 0
                        print(f"实际FPS: {30 / elapsed:.1f}")
                except cv2.error as cve:
                    print(f"摄像头帧处理有问题，记得看看{cve}")
                except Exception as re:
                    print(f"帧获取异常:{re}")
                    break
                QThread.msleep(10)
        except Exception as e:
            print(f"执行摄像头线程错误:{e}")
        finally:
            self.mutex.lock()
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self._running = False
            self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self._running = False
        self.mutex.unlock()
        if self.isRunning():
            self.wait(300)
        else:
            self.mutex.unlock()
        print("摄像头已释放")


class CvImageProvider(QQuickImageProvider):
    imageUpdated = Signal()
    recUpdated = Signal()
    recogAreaUpdated = Signal(str, int)
    recogReady = Signal(np.ndarray)  # 用于激活 main.py 中的识别流程

    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._current_image = QImage()
        self._bbox_image = QImage()
        self._recog_images = []
        self._recog_index = 0
        # self.model_pipe = TorchRecogPipe()
        self.current_frame = None

    def update_image(self, frame: np.ndarray) -> None:
        try:
            self.current_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cpy = frame_rgb.copy()
            h, w, ch = frame_cpy.shape
            q_image = QImage(frame_cpy.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self._current_image = q_image  # 确保QImage独立
            self._bbox_image = q_image.copy()
            self.imageUpdated.emit()
        except  Exception as e:
            print(f"在更新图像时发生错误: {e}")

    def update_image_from_path(self, file_path):
        local_path = QUrl(file_path).toLocalFile()
        print("local_path", local_path)
        img = cv2.imread(local_path, cv2.IMREAD_COLOR)
        self.current_frame = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cpy = img_rgb.copy()
        h, w, ch = img_cpy.shape
        self._current_image = QImage(img_cpy.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.imageUpdated.emit()

    def update_recog_image(self, frame: np.ndarray):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cpy = frame_rgb.copy()
            h, w, ch = frame_cpy.shape
            q_image = QImage(frame_cpy.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            # self.mutex.lock()
            self._current_image = q_image  # 确保QImage独立
            self.current_frame = frame_cpy.copy()
            # self.mutex.unlock()
            self.recUpdated.emit()
        except  Exception as e:
            print(f"在更新图像时发生错误: {e}")

    def update_recog_view(self, face_info: str, pic_path: str) -> None:
        try:  # 把 pic_path 转为绝对路径
            f_path = os.path.abspath(pic_path)
            f_img = cv2.imread(f_path, cv2.IMREAD_COLOR)
            f_img_rgb = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
            self._recog_images.append(f_img_rgb)
            self._recog_images.append(f_img)
            index = len(self._recog_images) - 1
            self.recogAreaUpdated.emit(face_info, index)
        except Exception as e:
            print(f"在发送图片到流式布局时发生错误: {e}")

    def requestImage(self, image_id, size, flags=0):
        """接收 QML 的图像请求并发送图像"""
        if image_id.startswith("recog?"):
            try:
                index = int(image_id.split("?")[1])
                if 0 <= index < len(self._recog_images):
                    img = self._recog_images[index]
                    h, w, ch = img.shape
                    img_emit = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                    return img_emit
            except Exception as e:
                print(f"向流式布局中发送图像时发生错误: {e}")
                return None
        elif image_id.startswith("current?"):
            try:
                return self._current_image  # self._current_image
            except Exception as e:
                print(f"向预览框发送图像时发生错误: {e}")
                return None
        else:
            try:
                return self._bbox_image
            except Exception as e:
                print(f"向预览框发送图像时发生错误: {e}")
                return None
