import QtQuick 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: card
    width: 200; height: 150
    color: "#EEEEEE"
    radius: 8
    ColumnLayout {
        anchors.fill: parent
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 100
            color: "#FFFFFF"
            border.color: "#DDDDDD"
            border.width: 2
            radius: 8
            Text {
                text: card.value
                anchors.centerIn: parent
                font.bold: true
                font.pixelSize: 28
                color: "#333333"
            }
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            color: "#F0F0F0"
            radius: 8
            Text {
                text: card.label
                anchors.centerIn: parent
                font.bold: true
                font.pixelSize: 28
                wrapMode: Text.WordWrap
                color: "#666666"
            }
        }
    }
    // 可绑定属性
    property string value
    property string label
}
