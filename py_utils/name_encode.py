#! /usr/bin/env python3
# coding: utf-8
"""
保存cv图片时将中文字符转化为易于保存的方式
"""
def name_encode(origin_name):
    """中文转化对应的编码字符
    转化为字符串
    :param origin_name: 中文名
    :return: 字符串
    """
    hexa_codes = []
    for char in origin_name:
        # 转unicode
        hexa_code = f"{ord(char):04X}"
        hexa_codes.append(hexa_code)
    result_name = ''.join(hexa_codes)
    return result_name

# 示例
if __name__ == "__main__":
    test_names = ["张三", "李娜", "王小明", "赵芳","lILY"]
    for name in test_names:
        encrypted = name_encode(name)
        print(f"原姓名: {name} → 加密结果: {encrypted}（长度: {len(encrypted)}）")
