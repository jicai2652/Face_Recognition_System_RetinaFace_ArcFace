#! /usr/bin/env python3
# coding: utf-8

"""
操作SQLite数据库,存放ArcFace输出的人脸特征向量,预期划分维度存入表中,暂时不分割特征向量
"""
import os
import sqlite3
import numpy as np


def create_db(db_path: str) -> None:
    """建立数据库"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 建表存放信息,暂存姓名,特征向量,图片路径, 后续应当修改
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS face_features
                       (
                           face_id      TEXT NOT NULL,
                           face_name    TEXT NOT NULL,
                           face_feature BLOB NOT NULL,
                           face_path    TEXT NOT NULL,
                           recog_times  INT  NOT NULL
                       )''')
        conn.commit()
        conn.close()
    except sqlite3.Error as se:
        print(f"创建数据库失败：{se}")


def add_msg_to_db(db_path: str,
                  face_id: str,
                  face_name: str,
                  face_feature: np.ndarray,
                  face_path: str) -> None:
    """信息注入"""
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute('''
                       INSERT INTO face_features (face_id, face_name, face_feature, face_path)
                       VALUES (?, ?, ?, ?)
                       ''', (face_id, face_name, face_feature.tobytes(), face_path))
        connection.commit()
        connection.close()
    except sqlite3.Error as se:
        print(f"用户{face_name}的特征信息存储异常：{se}")


def add_record_to_db(db_path: str, face_id: str, face_name: str, record_time: str) -> bool:
    """添加识别记录"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
                       UPDATE face_features
                       SET recog_times = recog_times + 1
                       WHERE face_id = ?
                         AND face_name = ?
                       ''', (face_id, face_name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as se:
        print(f"用户{face_name}的识别记录存储异常：{se}")
        return False

def delete_msg_from_db(db_path: str, face_id: str, face_name: str) -> bool:
    """信息删除"""
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute('''
                       SELECT face_path
                       FROM face_features
                       WHERE face_id = ?
                         AND face_name = ?
                       ''', (face_id, face_name))
        pic_path = cursor.fetchone()[0]
        if not pic_path:
            print(f"用户{face_name}信息删除失败")
            connection.close()
            return False
        cursor.execute('''
                       DELETE
                       FROM face_features
                       WHERE face_id = ?
                         AND face_name = ?
                       ''', (face_id, face_name))
        connection.commit()
        connection.close()
        abs_pic_path = os.path.abspath(pic_path)
        try:
            os.remove(abs_pic_path)
        except FileNotFoundError:
            print(f"用户{face_name}图片删除失败,请检查图片路径{abs_pic_path}")
            return False
        return True
    except sqlite3.Error as se:
        print(f"删除信息时数据库错误：{se}")
        return False


def get_total_users(db_path: str) -> int:
    """查询总人数"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT COUNT(*)
                       FROM face_features
                       ''')
        result = cursor.fetchone()[0]
        conn.close()
        return result
    except sqlite3.Error as se:
        print(f"查询总人数时数据库错误:{se}")
        return 0


def get_total_recog(db_path: str) -> int:
    """查询识别总次数"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT SUM(recog_times)
                       FROM face_features
                       ''')
        result = cursor.fetchone()[0]
        conn.close()
        return result
    except sqlite3.Error as se:
        print(f"查询识别总次数时数据库错误:{se}")
        return 0


def load_known(db_path: str) -> list[dict]:
    """加载编号，姓名，特征向量，图片路径,以用于识别界面"""
    try:
        if db_path is None:
            raise ValueError('Error: 请检查数据库文件')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT face_id, face_name, face_feature, face_path
                       FROM face_features
                       ''')
        rows = cursor.fetchall()
        conn.close()
        # 结果字典
        feature_db = []
        for row in rows:
            feature_db.append(
                {"face_id": row[0], "face_name": row[1], "face_feature": np.frombuffer(row[2], dtype=np.float32),
                 "face_path": row[3]})

        return feature_db
    except sqlite3.Error as se:
        print(f"加载特征时数据库错误:{se}")
        return []
    except Exception as e:
        print(f"加载特征时其他异常:{e}")
        return []


def load_records(db_path: str) -> list[dict]:
    """加载编号、姓名、识别次数，以便在管理页面展示"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT face_id, face_name, recog_times FROM face_features")
        rows = cursor.fetchall()
        conn.close()
        # 结果字典
        records = [{"face_id": [], "face_name": [], "recog_times": []}]
        for row in rows:
            records.append({
                "face_id": row[0],
                "face_name": row[1],
                "recog_times": row[2]})
        return records
    except sqlite3.Error as se:
        print(f"显示记录时数据库错误:{se}")
        return []


if __name__ == '__main__':
    db_path = '/database/face_data.db'
    result = load_known(db_path)
    if result is not None:
        for i in result:
            print(i)
