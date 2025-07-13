#! /usr/bin/env python3
# coding: utf-8
"""
相似度处理,参考自https://github.com/Justin-ljw/FaceMind/blob/main/face_process/face_recognize.py
"""
import sqlite3
import numpy as np
import torch


def cosine_similarity(known_faces: np.ndarray, current_face: np.ndarray, device: torch.device) -> np.ndarray:
    """计算cos相似度
    原项目为l1范数（曼哈顿距离），由于arcface训练的模型给出的向量各不相同，不便于推理，因此，选用余弦相似度进行判别
    :param known_faces: 已知的人脸特征
    :param current_face: 当前检测到的人脸特征
    :param device: torch设备
    :return: 计算得出的cos相似度
    """

    known_faces = torch.tensor(known_faces, dtype=torch.float32).to(device)
    current_face = torch.tensor(current_face, dtype=torch.float32).to(device)
    # 计算归一化范数
    known_norms = torch.linalg.norm(known_faces, dim=1, keepdim=True).reshape(-1)
    current_norm = torch.linalg.norm(current_face, dim=1, keepdim=True)

    # 计算相似度
    similarity = torch.matmul(known_faces, current_face.t()) / (known_norms * current_norm.t())

    return similarity.cpu().numpy()


# def detect_db_batch(known_features: np.ndarray, current_feature: np.ndarray, device: torch.device,
#                     batch_size: int = 1024, threshold: float = 0.6) -> tuple[int, float]:
#     """
#     分批处理已知特征与当前特征的相似度匹配
#     :param known_features: 已知特征数组
#     :param current_feature: 当前待匹配特征
#     :param device: 计算设备
#     :param batch_size: 每批处理大小
#     :param threshold: 相似度阈值
#     :return: (匹配索引, 最高相似度)
#     """
#     try:
#         face_index = -1
#         max_similarity_overall = 0
#
#         # 分批处理特征
#         for batch_start in range(0, len(known_features), batch_size):
#             batch_end = min(batch_start + batch_size, len(known_features))
#             batch_features = known_features[batch_start:batch_end]
#
#             # 计算当前批次相似度
#             similarities = cosine_similarity(batch_features, current_feature, device)
#             batch_max_sim = np.max(similarities)
#             batch_max_idx = int(np.argmax(similarities))
#
#             # 更新全局最高相似度
#             if batch_max_sim > max_similarity_overall:
#                 max_similarity_overall = batch_max_sim
#                 if batch_max_sim >= threshold:
#                     face_index = batch_start + batch_max_idx
#
#         # 输出匹配结果
#         if face_index != -1:
#             print(f"匹配成功：{face_index}，相似度：{max_similarity_overall:.4f}")
#         else:
#             print(f"未找到匹配，最高相似度：{max_similarity_overall:.4f}")
#
#         return face_index, max_similarity_overall
#
#     except Exception as e:
#         print(f"特征匹配出错: {e}")
#         return -1, 0

def detect_db_batch_muti(db_path: str, current_features: list[np.ndarray], device: torch.device,
                         batch_size: int = 1024, threshold: float = 0.6) -> list[tuple[str, str, float, str]]:
    """
    分批从数据库加载特征的批量匹配函数
    :param db_path: 数据库路径
    :param current_features: 当前检测到的多个人脸特征列表
    :param device: torch设备
    :param batch_size: 每批加载的特征数量
    :param threshold: 相似度阈值
    :return: 匹配结果列表，每个元素为(id, name, similarity, path)元组
    """
    # 初始化结果列表，存储每个当前特征的最佳匹配
    best_matches = [{
        'id': '',
        'name': '',
        'similarity': 0.0,
        'path': ''
    } for _ in current_features]

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 获取总特征数
        cursor.execute('SELECT COUNT(*) FROM face_features')
        total = cursor.fetchone()[0]
        if total == 0:
            print("数据库中没有已知人脸")
            return [('', '', 0.0, '')] * len(current_features)
        # 分批处理数据库特征, 每次读取batch_size条特征
        for db_i in range(0, total, batch_size):
            # 加载当前批次特征
            cursor.execute(
                'SELECT face_id, face_name, face_feature, face_path FROM face_features LIMIT ? OFFSET ?',
                (batch_size, db_i))
            batch_rows = cursor.fetchall()

            # 预处理当前批次
            batch_features = []
            batch_info = []
            for row in batch_rows:
                if row[1] is None:
                    continue
                batch_features.append(np.frombuffer(row[2], dtype=np.float32))
                batch_info.append((row[0], row[1], row[3]))

            # 转换为PyTorch张量
            db_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).to(device)
            db_norms = torch.norm(db_tensor, dim=1, keepdim=True)

            # 处理每个当前特征
            for i, feature in enumerate(current_features):
                current_tensor = torch.tensor(feature, dtype=torch.float32).to(device)
                current_norm = torch.norm(current_tensor, dim=1, keepdim=True)

                # 计算相似度
                similarity = torch.mm(db_tensor, current_tensor.t()) / (db_norms * current_norm.t())
                similarity = similarity.cpu().numpy().flatten()

                # 更新最佳匹配
                max_sim_idx = int(np.argmax(similarity))
                current_max_sim = similarity[max_sim_idx]

                if current_max_sim > best_matches[i]['similarity']:
                    best_matches[i] = {
                        'id': batch_info[max_sim_idx][0],
                        'name': batch_info[max_sim_idx][1],
                        'similarity': current_max_sim,
                        'path': batch_info[max_sim_idx][2]
                    }

        conn.close()

        # 准备最终结果
        final_results = []
        for match in best_matches:
            if match['similarity'] >= threshold:
                print(f"匹配成功：{match['name']}，相似度：{match['similarity']:.4f}")
                final_results.append((match['id'], match['name'], float(match['similarity']), match['path']))
            else:
                print("未找到匹配人脸")
                final_results.append(('', '', 0.0, ''))

    except Exception as e:
        print(f"分批匹配失败: {e}")
        final_results = [('', '', 0.0, '')] * len(current_features)

    return final_results


def detect_db_no_torch(db_path: str, current_features: list[np.ndarray],
                       threshold: float = 0.6) -> list[tuple[str, str, float, str]]:
    """
    不使用PyTorch的数据库匹配版本
    :param db_path: 数据库路径
    :param current_features: 当前检测到的多个人脸特征列表
    :param threshold: 相似度阈值
    :return: 匹配结果列表，每个元素为(id, name, similarity, path)元组
    """
    results = []
    try:
        # 加载数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT face_id, face_name, face_feature, face_path FROM face_features')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("数据库中没有已知人脸")
            return [('', '', 0.0, '')] * len(current_features)

        # 预处理数据库特征
        db_features = []
        db_info = []
        for row in rows:
            if row[1] is None:
                print("数据库中存在空值")
                continue
            db_features.append(np.frombuffer(row[2], dtype=np.float32))
            db_info.append((row[0], row[1], row[3]))

        db_array = np.array(db_features)
        db_norms = np.linalg.norm(db_array, axis=1, keepdims=True)

        # 处理每个当前特征
        for feature in current_features:
            feature = np.array(feature).reshape(1, -1)  # 确保是2D
            feature_norm = np.linalg.norm(feature, axis=1, keepdims=True)

            # 计算余弦相似度
            similarity = np.dot(db_array, feature.T) / (db_norms * feature_norm.T)
            similarity = similarity.flatten()

            # 找到最佳匹配
            max_sim_idx = int(np.argmax(similarity))
            max_similarity = similarity[max_sim_idx]

            if max_similarity >= threshold:
                match_id, match_name, match_path = db_info[max_sim_idx]
                print(f"匹配成功：{match_name}，相似度：{max_similarity:.4f}")
                results.append((match_id, match_name, float(max_similarity), match_path))
            else:
                print("未找到匹配人脸")
                results.append(('', '', 0.0, ''))

    except Exception as e:
        print(f"数据库中批量匹配特征失败: {e}")
        results = [('', '', 0.0, '')] * len(current_features)

    return results


def detect_db_batch_no_torch(db_path: str, current_features: list[np.ndarray],
                             batch_size=1024, threshold: float = 0.6) -> list[tuple[str, str, float, str]]:
    """
    不使用PyTorch的分批数据库匹配版本
    :param db_path: 数据库路径
    :param current_features: 当前检测到的多个人脸特征列表
    :param batch_size: 每批加载的特征数量
    :param threshold: 相似度阈值
    :return: 匹配结果列表，每个元素为(id, name, similarity, path)元组
    """
    # 初始化结果列表，存储每个当前特征的最佳匹配
    best_matches = [{
        'id': '',
        'name': '',
        'similarity': 0.0,
        'path': ''
    } for _ in current_features]

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取总特征数
        cursor.execute('SELECT COUNT(*) FROM face_features')
        total = cursor.fetchone()[0]
        if total == 0:
            print("数据库中没有已知人脸")
            return [('', '', 0.0, '')] * len(current_features)

        # 分批处理数据库特征
        for db_i in range(0, total, batch_size):
            # 加载当前批次特征
            cursor.execute(
                'SELECT face_id, face_name, face_feature, face_path FROM face_features LIMIT ? OFFSET ?',
                (batch_size, db_i))
            batch_rows = cursor.fetchall()

            # 预处理当前批次
            batch_features = []
            batch_info = []
            for row in batch_rows:
                if row[1] is None:
                    continue
                batch_features.append(np.frombuffer(row[2], dtype=np.float32))
                batch_info.append((row[0], row[1], row[3]))

            # 转换为numpy数组
            db_array = np.array(batch_features)
            db_norms = np.linalg.norm(db_array, axis=1, keepdims=True)

            # 处理每个当前特征
            for i, feature in enumerate(current_features):
                feature = np.array(feature).reshape(1, -1)
                feature_norm = np.linalg.norm(feature, axis=1, keepdims=True)

                # 计算相似度
                similarity = np.dot(db_array, feature.T) / (db_norms * feature_norm.T)
                similarity = similarity.flatten()

                # 更新最佳匹配
                max_sim_idx = int(np.argmax(similarity))
                current_max_sim = similarity[max_sim_idx]

                if current_max_sim > best_matches[i]['similarity']:
                    best_matches[i] = {
                        'id': batch_info[max_sim_idx][0],
                        'name': batch_info[max_sim_idx][1],
                        'similarity': current_max_sim,
                        'path': batch_info[max_sim_idx][2]
                    }

        conn.close()

        # 准备最终结果
        final_results = []
        for match in best_matches:
            if match['similarity'] >= threshold:
                print(f"匹配成功：{match['name']}，相似度：{match['similarity']:.4f}")
                final_results.append((match['id'], match['name'], float(match['similarity']), match['path']))
            else:
                print("未找到匹配人脸")
                final_results.append(('', '', 0.0, ''))

    except Exception as e:
        print(f"分批匹配失败: {e}")
        final_results = [('', '', 0.0, '')] * len(current_features)

    return final_results
