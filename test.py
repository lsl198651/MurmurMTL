# from utils.utils_dataset import get_id_position_org
from utils.util_saveInfo import *

if __name__ == "__main__":
    # output_path = r"D:\Shilong\new_murmur\02_dataset\02_4s_4k\npyFile_padded\organized_data"
    # data_path = r"D:\Shilong\new_murmur\02_dataset\02_4s_4k\npyFile_padded\index_files01_norm"
    # for root, dir, file in os.walk(data_path):
    #     for file in file:
    #         if file.endswith(".csv"):
    #             print('processing file:', file)
    #             get_id_position_org(root, output_path, file)
    #         # print('processing file:', file)
    #         # a = csv_to_dict(root, file)
    # root_path = r"E:\Shilong\02_dataset\00_5s_4k"

    # for k in range(5):
    #     for murmur_class in ['Absent', 'Present']:
    #         src_fold_path = root_path + r"\fold_" + str(k) + "\\" + murmur_class + "\\"
    #         target_dir = root_path + r'\fold_set_' + str(k)
    #         mkdir(target_dir)
    #         mkdir(target_dir + "\\absent\\")
    #         mkdir(target_dir + "\\present\\")
    #         for root, dir, file in os.walk(src_fold_path):
    #             for subfile in file:
    #                 files = os.path.join(root, subfile)
    #                 print(subfile)
    #                 state = subfile.split("_")[4]
    #                 if state == 'Absent':
    #                     shutil.copy(files, target_dir + "\\absent\\")
    #                 else :
    #                     shutil.copy(files, target_dir + "\\present\\")
    # for k in range(5):
    #     src_fold_root_path = root_path + r"'\fold_set_" + str(k)
    #     for murmur_class in ['absent', 'present']:
    #         src_fold_path = src_fold_root_path + "\\" + murmur_class + "\\"

    # path=r'E:\Shilong\02_dataset\01_5s_4k_txt\fold_2\Present\14241_AV\14241_AV_5s_0_Early-systolic+nan_None_210.txt'
    # a=read_txt(path)
    # print(a)
    path = r'E:\Shilong\02_dataset\05_5s_4k_txt\npyFile_padded\npy_files01_norm\absent_tags_norm01_fold1.npy'
    a = np.load(path, allow_pickle=True)
    print(a)
