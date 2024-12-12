def save_as_txt(text_to_save, file_path):
    # 类型检查
    if not isinstance(text_to_save, str):
        text_to_save = str(text_to_save)[1:-1]
        text_to_save=text_to_save.replace("\n"," ")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_to_save)
    # print(f"字符串已保存到文件：{file_path}")


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()