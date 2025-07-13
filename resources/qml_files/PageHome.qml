import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

Rectangle {
    id: homePage
    width: parent.width
    height: parent.height
    color: "#DDDDDD"
    radius: 8
    clip:true
    Image {
        anchors.fill: parent
        source:"qrc:/homepage.jpg"
        fillMode: Image.PreserveAspectCrop
    }
    function initPage() {
        console.log("激活主界面", signalBike)
        controller.update_total_users()
        controller.update_today_recog()
        controller.update_total_recog()
        controller.update_device()
        controller.update_documents()
    }

    Component.onCompleted: {homePage.initPage()}
    // 状态卡片格栅
    Grid {
        id: grid
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        columns: 3; columnSpacing: 20
        rows: 2; rowSpacing: 10
        // 设置Grid距离边界的距离
        padding: 20
        // 信息卡片
        CardStatus {
            id: totalUsers
            value: "0"
            label: "当前注册人数"
        }
        CardStatus {
            id: todayRecog
            value: "0"
            label: "当前识别次数"
        }
        CardStatus {
            id: totalRecog
            value: "0"
            label: "累计识别次数"
        }
        CardStatus {
            id: nowDevice
            value: "CPU"
            label: "所用计算设备"
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    console.log("尝试切换计算设备")
                    controller.check_devices()
                }
            }
        }
        CardStatus {
            id: configPath
            value: "det_cfg.toml"
            label: "运行配置"
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    console.log("尝试打开配置文件:", "../" + configPath.value)
                    Qt.openUrlExternally("../" + "det_cfg.toml")// configPath.value)
                }
            }
        }
        CardStatus {
            id: guidePath
            value: "README.md"
            label: "使用教程"
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    console.log("尝试打开使用教程:", "../../" + guidePath.value)
                    Qt.openUrlExternally("../../" + guidePath.value)
                }
            }
        }
        Connections {
            target: signalBike

            function onTotalUsersChanged(value) {totalUsers.value = value}
            function onTodayRecogChanged(value) {todayRecog.value = value}
            function onTotalRecogChanged(value) {totalRecog.value = value}
            function onNowDeviceChanged(value) {nowDevice.value = value; console.log("当前计算设备:", value)}
            function onFilePathChanged(value1,  value2) {configPath.value = value1; guidePath.value = value2}
            // function onGuidePathChanged(value) {guidePath.value = value}
        }
    }
}