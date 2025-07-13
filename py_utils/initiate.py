import os

import tomli

from py_utils.db_manager import create_db

def ini_check(db_path):#, r_path, a_path):
    if not os.path.exists(db_path):
        create_db(db_path)


def load_mobilenet_config(config_path):
    """
    从 TOML 文件中加载 mobilenet0_25 的配置。

    :param config_path: TOML 配置文件路径
    :return: 包含 mobilenet0_25 配置的字典
    """
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    # 提取 mobilenet0_25 的配置
    mobilenet_config = config.get("retinamodel")
    if not mobilenet_config:
        raise ValueError("mobilenet0_25 configuration not found in the TOML file.")

    return mobilenet_config


def load_path(config_path: str) -> dict:
    """
    根据cfg.toml中的frame_version配置，返回对应的模型路径

    :param config_path: TOML配置文件路径
    :return: 包含retina和arcface模型路径的字典
    """
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    # 获取当前使用的框架版本
    frame_config = config.get("frame_version", {})

    if frame_config.get("pytorch", False):
        retina_pt_abspath = os.path.abspath(config["model_path"]["retina_torch"])
        arcface_pt_abspath = os.path.abspath(config["model_path"]["arcface_torch"])
        return {
            "version": "pytorch",
            "retina": retina_pt_abspath,
            "arcface": arcface_pt_abspath
        }
    elif frame_config.get("onnx", False):
        retina_onnx_abspath = os.path.abspath(config["model_path"]["retina_onnx"])
        arcface_onnx_abspath = os.path.abspath(config["model_path"]["arcface_onnx"])
        return {
            "version": "onnx",
            "retina": retina_onnx_abspath,
            "arcface": arcface_onnx_abspath
        }
    elif frame_config.get("ascend", False):
        retina_om_abspath = os.path.abspath(config["model_path"]["retina_om"])
        arcface_om_abspath = os.path.abspath(config["model_path"]["arcface_om"])
        return {
            "version": "ascend",
            "retina": retina_om_abspath,
            "arcface": arcface_om_abspath
        }
    else:
        print(f"initiate.py提示：\n\t没有启用中的框架与模型，请检查{config_path}文件中的[frame_version]配置")   # raise会阻止下面的return，因此直接print
        return {"version": "未配置"}

cfg_mnet=load_mobilenet_config(os.path.join(os.path.dirname(__file__), "..", "resources", "cfg.toml"))
# 示例用法
if __name__ == "__main__":
    # config_path = "/Users/wuchenglei/PycharmProjects/ra_face/FaceDetectionRecognition/data/model_cfg.toml"
    config_path = os.path.join(os.path.dirname(__file__), "..", "resources", "cfg.toml")

    print(config_path)
    cfg_mnet = load_mobilenet_config(config_path)
    print(cfg_mnet)

    # 测试load_path函数
    model_paths = load_path(config_path)
    print(f"当前模型路径: {model_paths}")
    print(model_paths["retina"])
