import QtQuick 2.15


Rectangle {
    // 点击区域时,发送一个信号以控制
    signal btnClicked()
    property string textStr: ""
    property int fontSize: 20
    property string enterColor: "#777777"
    property string leaveColor: "#444444"

    width: 40; height: 30
    radius: 8
    color: leaveColor
    // color: "#444444"
    border.color: "transparent"

    Text {
        text: textStr
        color: "white"
        font.pointSize: fontSize
        font.bold: true
        anchors.centerIn: parent
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onClicked: {
            parent.btnClicked()
        }
        onEntered: parent.color = enterColor
        onExited: parent.color = leaveColor
    }
}