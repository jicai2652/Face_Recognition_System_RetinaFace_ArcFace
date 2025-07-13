//PageFaceRecog.qml
import QtQuick
import QtQuick.Layouts

Rectangle {
    width: parent.width
    height: parent.height
    color:"#DDDDDD"
    radius: 8

    RowLayout {
        anchors.fill: parent
        // 摄像头预览框
        Rectangle {
            width: 640
            height: 480
            radius: 8
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "black"
            Image {
                id: recImage
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                source: ""
                cache: false
                Connections {
                    target: cvProvider
                    function onRecUpdated() {
                        recImage.source = "image://cvProvider/current?" + Date.now()
                    }
                }
            }
        }
        // 控制按钮列
        ColumnLayout {
            spacing: 10
            Layout.alignment: Qt.AlignTop
            Layout.preferredWidth: 200
            Layout.fillWidth: true
            MyButton {
                id: recPlayButton
                Layout.fillWidth: true
                height: 40
                radius: 8
                property bool isCameraActive: false
                textStr: recPlayButton.isCameraActive ? "关闭摄像头" : "打开摄像头"
                onBtnClicked: {
                    // controller.camera_trigger("rec")
                    recPlayButton.isCameraActive = !recPlayButton.isCameraActive
                    if (isCameraActive){
                        controller.camera_trigger("rec")
                        // recImage.source = "image://cvProvider/rec?" + Date.now()
                        recPlayButton.textStr = "关闭摄像头"
                        console.log("识别页面正在使用摄像头")
                    }else{
                        controller.camera_release()
                        recPlayButton.textStr = "打开摄像头"
                    }


                }
            }
            MyButton {
                id: detectButton
                Layout.fillWidth: true
                height: 40
                textStr: "开始识别"
                onBtnClicked: controller.recog_start()
            }
            MyButton {
                id: clearButton
                Layout.fillWidth: true
                height: 40
                radius: 8
                textStr: "清空预览框"
                onBtnClicked: {
                    recImage.source = ""
                    recogModel.clear()
                }
            }
            // 流式布局回显
            Rectangle {
                id: recogArea
                radius: 8
                color: "#777777"
                Layout.fillHeight: true
                Layout.fillWidth: true
                border.width: 5; border.color: "#999999"
                Text {
                    text: "识别结果(流布局）"
                    font.pointSize: 20
                    font.bold: true
                    anchors.centerIn: parent
                }
                Flow {
                    id: recogFlow
                    spacing: 10
                    anchors {fill: parent; margins: 10}
                    ListModel {
                        id: recogModel
                    }
                    Connections {
                        target: cvProvider
                        function onRecogAreaUpdated(faceInfo, index) {
                            console.log("onRecogAreaUpdated: ", faceInfo, index)
                            recogModel.append({"faceInfo": faceInfo, "index": index})
                        }
                    }
                    add: Transition {
                        NumberAnimation {
                            property: "opacity"
                            from:0
                            to: 1
                            duration: 1000
                            easing.type: Easing.InCubic
                        }
                    }
                    Repeater {
                        model: recogModel
                        delegate: Rectangle {
                            width: 112
                            height: 140
                            color: "#999999"
                            Column {
                                spacing: 5
                                anchors.fill: parent
                                Image {
                                    width: 112
                                    height: 112
                                    source: "image://cvProvider/recog?" + model.index
                                    fillMode: Image.PreserveAspectFit
                                }
                                Text {
                                    text: model.faceInfo
                                    width: parent.width
                                    horizontalAlignment: Text.AlignHCenter
                                    color: "white"
                                    font.pixelSize: 16
                                    elide: Text.ElideLeft
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}