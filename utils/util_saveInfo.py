import numpy as np


def save_as_txt(text_to_save, file_path):
    # 类型检查
    if not isinstance(text_to_save, str):
        text_to_save = str(text_to_save)[1:-1]
        text_to_save = text_to_save.replace("\n", " ")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_to_save)
    # print(f"字符串已保存到文件：{file_path}")


def save_as_txt_np(text_to_save, file_path):
    # 类型检查
    # if not isinstance(text_to_save, str):
    #     text_to_save = str(text_to_save)[1:-1]
    #     text_to_save = text_to_save.replace("\n", " ")
    # with open(file_path, "w", encoding="utf-8") as file:
    #     file.write(text_to_save)
    np.savetxt(file_path, text_to_save, fmt='%d', delimiter=' ')
    # print(f"字符串已保存到文件：{file_path}")


# 输出为数组形式
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        arrays = file.read()
        num_array = np.array(list(map(int, arrays.split())))
        return num_array


def read_txt_np(file_path):
    return np.loadtxt(file_path, dtype=int, delimiter=' ')
