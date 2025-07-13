import torch
import arcface
from nets.arcface import Arcface as arcface
import onnxruntime as ort
import numpy as np


def convert2onnx_demo():
    model_path = './logs/acc_9737.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading weights into state dict...')
    net = arcface(backbone='iresnet50', mode="predict").eval()
    state_dict = torch.load(model_path, map_location=device)
    # missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=True)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    net = net.to(device)
    batch_size = 4
    print('{} model loaded.'.format(model_path))
    dummy_input = torch.randn(batch_size, 3, 112, 112).to(device)
    # 导出路径
    onnx_path = './model_data/acc_9737.onnx'

    input_names = ["input"]  # 自定义输入名
    output_names = ["output"]  # 自定义输出名
    dynamic_axes = {
        "input": {0: "batch_size"},  # 使用自定义输入名
        "output": {0: "batch_size"}
    }
    opset = 12
    torch.onnx.export(net, dummy_input, onnx_path, input_names=input_names,
    output_names=output_names,opset_version=opset, dynamic_axes=dynamic_axes, do_constant_folding=True)

    ort_session = ort.InferenceSession(onnx_path)
    # outputs = ort_session.run(None, {'images': np.random.randn(batch_size, 3, 112, 112).astype(np.float32)})
    print([inp.debugName() for inp in torch.jit.trace(net, dummy_input).graph.inputs()])

    outputs = ort_session.run(None, {'input': np.random.randn(batch_size, 3, 112, 112).astype(np.float32)})  # 使用正确的输入名
    print(outputs[0], outputs[0].shape)


convert2onnx_demo()