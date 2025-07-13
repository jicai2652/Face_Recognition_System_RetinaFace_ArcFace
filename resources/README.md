# 运行资源
## 描述
存放资源文件,包括: QML文档, 图片, 字体, 模型等

## 目录结构
### fonts: 存放字体文件
```
fonts/
  - HarmonyOS_SansSC_Medium.ttf
```
### icons: 存放图标文件
```
icons/
  - db_manage.png       :  数据库管理页面按钮
  - face_enroll.png     : 人脸录入页面按钮
  - face_recog.png      : 人脸识别页面按钮
  - home_page.png       : 首页按钮
  - logo.png            : 程序图标，作者: 天野岁 来源：https://weibo.com/2809866727/PpOo4CWXN
```
### models: 存放模型文件
```
models/
  - acc_9737.onnx            : ArcFace模型onnx文件(iResNet50)
  - acc_9737.pth             : ArcFace模型pth文件(iResNet50)
  - mobilenet0.25_Final.onnx : 预训练RetinaFace模型onnx文件
  - mobilenet0.25_Final.pth  : 预训练RetinaFace模型pth文件
```
### pages: 存放qml页面文件
```
pages/
 - CardStatus.qml   : 存放自定义状态卡片文件
 - MyButton.qml     :  存放人脸录入页面文件
 - PageDBManage.qml     : 存放数据库管理页面文件
 - PageFaceEnroll.qml   : 存放人脸录入页面文件
 - PageFaceRecog.qml    : 存放人脸识别页面文件
 - PageHome.qml     : 存放主界面文件
```
### main_window.qml: 程序界面
homepage.jpg: 首页图片  
resources.qrc: 存放资源文件