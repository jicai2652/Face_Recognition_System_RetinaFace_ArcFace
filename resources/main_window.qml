import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Dialogs
import "qml_files"
// 主体界面
ApplicationWindow {
    id: root
    visible: true
    width: 1200
    height: 800
    color: "transparent" // 透明背景方便定义圆角
    flags: Qt.Window | Qt.FramelessWindowHint
    font: Qt.application.font

    signal image_path(string filePath)

    onClosing: {controller.camera_release()}
    property bool image_ready: false
    //文件选择对话框
    FileDialog {
        id: file_choose
        title: "选择图片文件"
        nameFilters: ["图片文件(*.png *.jpg * jpeg)"]
        onAccepted: {
            image_path(file_choose.selectedFile.toString())
            root.image_ready = false
            root.image_ready = true
        }
    }
    // 窗口拖动处理
    MouseArea {
        anchors.fill: parent
        property point clickPos: "0,0"
        onPressed: (mouse) => {clickPos = Qt.point(mouse.x, mouse.y)}
        onPositionChanged: (mouse) => {
            let delta = Qt.point(mouse.x - clickPos.x, mouse.y - clickPos.y)
            root.x += delta.x
            root.y += delta.y
        }
    }
    // 显示内容
    Rectangle {
        id: background
        width: parent.width
        height: parent.height
        radius: 8
        color: "#F5F5F5"
        // 顶部标题栏: 三按钮
        Rectangle {
            width: parent.width
            height: 50
            radius: 8
            color: "#333333"
            Text {
                text: "基于深度学习的人脸识别系统"
                color: "#FFFFFF"
                font.pointSize: 24
                font.bold: true
                anchors.centerIn: parent
            }
            /// 窗口控制按钮
            RowLayout {
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: parent.verticalCenter
                spacing: 10
                // 最小化按钮
                MyButton {
                    id: minimizeButton
                    textStr: "—"
                    fontSize: 20
                    onBtnClicked: root.showMinimized()
                }
                // 全屏切换按钮
                MyButton {
                    id: maximizeButton
                    textStr: "囗"// "□"
                    onBtnClicked: root.visibility === ApplicationWindow.Windowed ? root.showFullScreen() : root.showNormal()
                }
                // 关闭窗口按钮
                MyButton {
                    id: closeButton
                    textStr: "X"
                    enterColor: "#FFAAAA"
                    leaveColor: "#FF5555"
                    onBtnClicked: {
                        controller.camera_release()
                        root.close()
                    }
                }
            }
        }
        // 主内容区域
        RowLayout {
            anchors.top: parent.top
            anchors.topMargin: 70
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 20
            spacing: 10
            // 左侧导航栏
            ColumnLayout {
                id: navColumn
                Layout.preferredWidth: 90
                Layout.alignment: Qt.AlignCenter
                spacing: 20
                // 页面索引
                property int currentIndex: 0
                // 图标源列表
                property var iconSources: ["qrc:/icons/home_page.png", "qrc:/icons/face_enroll.png",
                    "qrc:/icons/face_recog.png", "qrc:/icons/db_manage.png"]
                // 导航按钮样式
                Repeater {
                    model: ["首页", "录入", "识别", "管理"]
                    delegate: Rectangle {
                        width: 80
                        height: 110
                        radius: 8
                        color: navColumn.currentIndex === index ? "#4CAF50" : "#444444"
                        Image {
                            anchors.top: parent.top
                            source: navColumn.iconSources[index]
                            fillMode: Image.PreserveAspectCrop
                            clip: true
                        }
                        Text {
                            anchors.bottom: parent.bottom
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: modelData
                            color: "white"
                            font.pointSize: 24
                        }
                        MouseArea {
                            anchors.fill: parent
                            onClicked: {
                                swipeView.currentIndex = index
                                navColumn.currentIndex = index
                            }
                        }
                        Behavior on color {
                            ColorAnimation {
                                duration: 200
                            }
                        }
                    }
                }
            }
            // 右侧内容区域,Item承载空间更能节约性能
            Item {
                id: pageContainer
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                // 页面切换控制：来不及改写成Loader，为了保障直观易用选用平滑视图
                SwipeView {
                    id: swipeView
                    anchors.fill: parent
                    orientation: Qt.Vertical // 设置为纵向滑动
                    currentIndex: navColumn.currentIndex
                    // 监听页面切换
                    onCurrentIndexChanged: {
                        controller.camera_release()
                        if (currentIndex === 0) {
                            console.log("切换到首页")
                            homePage.initPage()
                        } else if (currentIndex === 1) {
                            console.log("切换到录入页面")
                        } else if (currentIndex === 2) {
                            console.log("切换到识别页面")
                        } else if (currentIndex === 3) {
                            console.log("切换到管理页面")
                        }
                    }
                    // 四页面, 用Item包裹,以控制大小填充,Item需隐式控制宽高
                    Item {
                        id: pageZero
                        PageHome {
                            id: homePage
                            anchors.fill: parent
                            color: "#DDDDDD"
                            radius: 8
                        }
                    }
                    Item {
                        id: pageOne
                        PageFaceEnroll {
                            id: faceEnrollPage
                            anchors.fill: parent
                            color: "#DDDDDD"
                            radius: 8
                        }
                    }
                    Item {
                        id: pageTwo
                        PageFaceRecog {
                            id: faceRecogPage
                            anchors.fill: parent
                            color: "#DDDDDD"
                            radius: 8
                        }
                    }
                    Item {
                        id: pageThree
                        PageDBManage {
                            id: dbManagePage
                            anchors.fill: parent
                            color: "#DDDDDD"
                            radius: 8
                        }
                    }
                }
            }
        }
    }
    Component.onCompleted: {
        console.log("页面已加载，开始加载模型")
        controller.check_devices()
    }
}
