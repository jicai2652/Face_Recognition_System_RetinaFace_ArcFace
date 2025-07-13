#! /usr/bin/python3
# coding: utf-8
"""
模型加载工具模块
主要用于RetinaFace和ArcFace模型的权重加载
参考实现来自：https://github.com/biubug6/Pytorch_Retinaface/blob/master/convert_to_onnx.py
"""
import torch


def check_keys(model, pretrained_state_dict):
    """检查模型权重键值匹配

    :param model: 要加载权重的模型
    :param pretrained_state_dict: 预训练权重字典
    :return: 检查是否通过
    :raises AssertionError: 当没有匹配的权重键时抛出异常
    """
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, '从预训练检查点没有加载任何权重'
    return True


def remove_prefix(state_dict, prefix):
    """移除权重键名前缀
    旧式模型参数名通常带有'module.'前缀，此函数用于移除该前缀

    :param state_dict: 模型权重字典
    :param prefix: 要移除的前缀(通常是'module.')
    :return: 移除前缀后的权重字典
    """
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

# def load_model(model:torch.nn.Module, pretrained_path:str, load_to_cpu: bool)-> torch.nn.Module:
#     """加载预训练权重
#     :param model: 目标模型
#     :param pretrained_path: 预训练权重文件
#     :param load_to_cpu: 是否加载到cpu
#     :return: 加载了预训练权重后的模型
#     """
#     print('Loading pretrained model from {}'.format(pretrained_path))
#     if load_to_cpu:
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
#     else:
#         device = torch.cuda.current_device()
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
#     # 兼容旧版本权重文件
#     if "state_dict" in pretrained_dict.keys():
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#     else:
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#     check_keys(model, pretrained_dict)
#     #  加载权重
#     model.load_state_dict(pretrained_dict, strict=False)
#     return model
def load_model(model:torch.nn.Module, pretrained_path:str, now_device: torch.device)-> torch.nn.Module:
    """加载预训练权重

    :param model: 目标模型
    :param pretrained_path: 预训练权重文件
    :param now_device: 目标设备
    :return: 加载了预训练权重后的模型
    """
    print(f"RetinaFace模型已加载到设备{now_device}:\t{pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location=now_device)
    # 兼容旧版本权重文件
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    #  加载权重
    model.load_state_dict(pretrained_dict, strict=False)
    return model
