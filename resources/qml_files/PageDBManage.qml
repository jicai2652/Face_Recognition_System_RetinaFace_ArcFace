import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.qmlmodels

Rectangle {
    width: parent.width
    height: parent.height
    color:"#DDDDDD"
    radius: 8
    Component.onCompleted: {
        controller.load_table_db()
    }
    ColumnLayout {
        spacing: 20
        anchors.fill: parent
        anchors.margins: 15
        Layout.alignment: Qt.AlignTop
        Label {
            text: "用户数据管理"
            Layout.alignment: Qt.AlignHCenter
            font {bold: true; pixelSize: 22}
        }
        ListModel {
            id: dbTableList
        }
        Connections {
            target: signalBike
            function onTableDBChanged(records){
                dbTableList.clear()
                for (let i=1; i < records.length; i++) {
                    dbTableList.append({
                        "uid": records[i].face_id,
                        "name": records[i].face_name,
                        "recogCount": records[i].recog_times
                    })
                }
                console.log("已更新次数统计")
            }
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 8
            border {width: 1; color: "#CCCCCC"}
            Column {
                anchors.fill: parent
                spacing: 1
                //表头
                Row {
                    width: parent.width
                    height: 50
                    spacing: 0
                    Repeater {
                        model: ["编号", "姓名", "识别次数"]
                        Rectangle {
                            width: {index === 0 ? parent.width * 0.2 : index === 1 ? parent.width * 0.3 : parent.width * 0.3}
                            height: 50
                            color: "#f3f3f3"
                            radius: 6
                            border { width: 1; color: "#cccccc" }

                            Text {
                                text: modelData
                                anchors.centerIn: parent
                                font.pixelSize: 16
                            }
                        }
                    }

                    MyButton {
                        width: parent.width * 0.2
                        height: 50
                        textStr: "刷新"
                        enterColor: "#AAAAAA"
                        leaveColor: "#777777"
                        onBtnClicked: {
                            console.log("数据库表格视图刷新")
                            controller.load_table_db()
                        }
                    }
                }
                //表格内容
                TableView {
                    id: tableView
                    width: parent.width
                    height: parent.height - 51
                    model: dbTableList
                    clip: true
                    delegate: Item {
                        implicitWidth: tableView.width //代理方式创建：需要隐式设置宽高
                        implicitHeight: 50
                        Row {
                            width: tableView.width
                            height: 50
                            spacing: 0
                            TableCell { width: tableView.width * 0.2; text: uid}
                            TableCell { width: tableView.width * 0.3; text: name}
                            TableCell { width: tableView.width * 0.3; text: recogCount }
                            TableCell {
                                width: tableView.width * 0.2;
                                Text {
                                    text: "删除"
                                    color: "#0078d4"
                                    font {pixelSize: 16; underline: true}
                                    anchors.centerIn: parent
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        console.log("此时删除数据库中对应信息")
                                        controller.delete_user(uid, name)
                                        dbTableList.remove(index)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    component TableCell : Rectangle {
        width: 100
        height: 50
        radius: 6
        color: "white"
        border { width: 1; color: "#cccccc" }
        property alias text: label.text
        Text {
            id: label
            anchors.centerIn: parent
            font.pixelSize: 16
        }
    }
}