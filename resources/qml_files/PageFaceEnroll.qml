//PageFaceRecog.qml
import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls.Basic 2.15 // 用Basic样式->默认样式不准修改对话框
import QtQuick.Dialogs

Rectangle {
    width: parent.width
    height: parent.height
    color: "#DDDDDD"
    radius: 8

    RowLayout {
        anchors.fill: parent
        // 显示区域
        Rectangle {
            id: viewPlace
            width: 640
            height: 480
            radius: 8
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "black"
            // 摄像头预览框
            Image {
                id: rawImage
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                source: ""
                cache: false
                Connections {
                    target: cvProvider
                    function onImageUpdated() {
                        rawImage.source = "image://cvProvider/current?" + Date.now()
                    }
                }
            }
        }
        // 控制按钮列
        ColumnLayout {
            spacing: 10
            Layout.alignment: Qt.AlignCenter
            Layout.preferredWidth: 200
            Layout.fillHeight: true
            // 输入框
            RowLayout {
                Layout.fillWidth: true
                height: 50
                spacing: 5
                Rectangle {
                    width: 80
                    height: 50
                    radius: 8
                    color: "#BBBBBB"
                    Label {
                        text: "编号："
                        color: "#000000"
                        font.pointSize: 20
                        font.bold: true
                        anchors.centerIn: parent
                    }
                }
                TextField {
                    id: userId
                    placeholderText: "请输入ID编号"
                    Layout.fillWidth: true
                    height: 50
                    color: "#C3C3C3"
                    font.pointSize: 20
                    font.bold: true
                    verticalAlignment: Text.AlignVCenter
                    background: Rectangle {
                        implicitWidth: parent.width
                        implicitHeight: parent.height
                        radius: 8
                        border.width: 2
                        border.color: "#ccccff"
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true
                height: 50
                spacing: 5
                Rectangle {
                    width: 80; height: 50
                    radius: 8
                    color: "#BBBBBB"
                    Label {
                        text: "姓名："
                        color: "#000000"
                        font.pointSize: 20
                        font.bold: true
                        anchors.centerIn: parent
                    }
                }
                TextField {
                    id: userName
                    placeholderText: "请输入姓名"
                    Layout.fillWidth: true
                    height: 50
                    color: "#C3C3C3"
                    font.pointSize: 20
                    font.bold: true
                    verticalAlignment: Text.AlignVCenter
                    background: Rectangle {
                        implicitWidth: parent.width
                        implicitHeight: parent.height
                        radius: 8
                        border.width: 2
                        border.color: "#ccccff"
                    }
                }
            }
            MyButton {
                id: selectImage
                Layout.fillWidth: true
                height: 40
                textStr: "选择图片"
                onBtnClicked: file_choose.open()
            }
            MyButton {
                id: rawPlayButton
                Layout.fillWidth: true
                height: 40
                property bool isCameraActive: false
                textStr: "打开摄像头"
                onBtnClicked: {
                    controller.camera_trigger("raw")
                    rawPlayButton.isCameraActive = !rawPlayButton.isCameraActive
                    if (rawPlayButton.isCameraActive) {
                        rawPlayButton.textStr = "关闭摄像头"
                        console.log("录入页面正在使用摄像头")
                    } else {
                        controller.camera_release()
                        rawPlayButton.textStr = "打开摄像头"
                    }
                }
            }
            MyButton {
                id: clearButton
                Layout.fillWidth: true
                height: 40
                textStr: "清空预览框"
                onBtnClicked: rawImage.source = ""
            }
            MyButton {
                id: enrollButton
                Layout.fillWidth: true
                height: 40
                textStr: "录入信息"
                onBtnClicked: controller.take_picture(userId.text, userName.text)
            }
        }
    }
}