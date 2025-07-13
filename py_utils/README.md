# 软件包
## 描述
除主程序main.py外的python文件均应位于此包,以便调试与管理
## 目录结构

> Py_codes/SQLite/db_manage.py

## 目录结构

### cv_utils:
```
cv_utils/
  - align_trans.py  : 人脸对齐代码
  - det_convert.py  : 预处理及后处理操作代码
  - py_cpu_nms.py   : NMS代码
```
### qt_utils:
```
qt_utils/
  - compiled_resources.py   : 预编译的资源文件
  - image_provider.py       : 摄像头线程控制，及QtQuick界面图像源控制
```
### torch_utils:
```
torch_utils/
  - nets/       :网络模型基础文件
  -  - arc_net.py   :  ArcFace网络模型
  -  - iresnet.py   :  iResNet网络模型
  -  - retina_net.py    :  RetinaFace网络模型
```
### 其他操作：
```
./
 - db_manager.py        : 数据库操作
 - initiate.py          : 程序初始化
 - name_encode.py       : 简单加密（utf-8 Hexa转换)
 - recog_progress.py    : 封装后的识别流程
```
