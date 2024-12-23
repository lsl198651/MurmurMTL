import datetime
import logging
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix

from utils.util_dataset import csv_reader_row, csv_reader_cl, csv_to_dict


def logger_init(
        log_level=logging.DEBUG,
        log_dir=r"./log"):
    """
       初始化logging
    """
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 指定日志格式
    date = datetime.now()
    log_path = os.path.join(log_dir, str(datetime.now().strftime("%Y-%m%d %H%M")) + ".log")
    formatter = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        level=log_level,
        format=formatter,
        datefmt="%m%d %H%M",
        handlers=[logging.FileHandler(log_path, encoding='utf-8', mode="a"), logging.StreamHandler(sys.stdout)])
    logging.disable(logging.DEBUG)


def new_segment_cluster(present_result_list: list, test_fold: list, set_path: str):
    global present_id
    root_path = fr"{set_path}\npyFile_padded\organized_data"
    all_result_dic = {}
    for k in test_fold:
        present_result_set = set(present_result_list)
        file_absent = root_path + rf"\organized_data_fold{k}_absent_disc.csv"
        # print('processing file:', file_absent)
        absent_test_dic = csv_to_dict(file_absent)
        file_present = root_path + rf"\organized_data_fold{k}_present_disc.csv"
        present_test_dic = csv_to_dict(file_present)
        test_dic = absent_test_dic
        for test_dic_key, _ in present_test_dic.items():
            if test_dic_key in test_dic.keys():
                for sub_key, _ in present_test_dic[test_dic_key].items():
                    test_dic[test_dic_key][sub_key] = present_test_dic[test_dic_key][sub_key]
            else:
                test_dic[test_dic_key] = present_test_dic[test_dic_key]

        present_csv_name = rf"{set_path}\present_fold_{k}.csv"
        present_id = csv_reader_cl(present_csv_name, 0)

        # test_dic = {**absent_test_dic, **present_test_dic}
        for test_dic_key, _ in test_dic.items():
            # print(test_dic_key)
            all_result_dic[test_dic_key] = {}
            for key, index_list in test_dic[test_dic_key].items():
                all_result_dic[test_dic_key][key] = 1 if (len(
                    [element for element in present_result_set if element in index_list])) / len(
                    index_list) > 0.5 else 0

    for key, pos_res in all_result_dic.items():
        patient_res = 0
        for _, res in pos_res.items():
            if res:
                patient_res = 1
                break
        all_result_dic[key]['result'] = patient_res
        if key in present_id:
            all_result_dic[key]['target'] = 1
        else:
            all_result_dic[key]['target'] = 0
    result = []
    target = []
    for key, value in all_result_dic.items():
        result.append(value['result'])
        target.append(value['target'])
    # print('done')
    return result, target


def new_duration_cluster(present_result_list: list, test_fold: list, set_path: str):
    global present_id
    root_path = fr"{set_path}\npyFile_padded\organized_data"
    all_result_dic = {}
    for k in test_fold:
        present_result_set = set(present_result_list)
        file_absent = root_path + rf"\organized_data_fold{k}_absent_disc.csv"
        # print('processing file:', file_absent)
        absent_test_dic = csv_to_dict(file_absent)
        file_present = root_path + rf"\organized_data_fold{k}_present_disc.csv"
        present_test_dic = csv_to_dict(file_present)
        test_dic = absent_test_dic
        for present_test_dic_key, _ in present_test_dic.items():
            if present_test_dic_key in test_dic.keys():
                for sub_key, _ in present_test_dic[present_test_dic_key].items():
                    if sub_key in test_dic[present_test_dic_key].keys():
                        test_dic[present_test_dic_key][sub_key].extend(present_test_dic[present_test_dic_key][sub_key])
                    else:
                        test_dic[present_test_dic_key][sub_key] = present_test_dic[present_test_dic_key][sub_key]
            else:
                test_dic[present_test_dic_key] = present_test_dic[present_test_dic_key]

        present_csv_name = rf"{set_path}\present_fold_{k}.csv"
        present_id = csv_reader_cl(present_csv_name, 0)

        # test_dic = {**absent_test_dic, **present_test_dic}
        for present_test_dic_key, _ in test_dic.items():
            # print(test_dic_key)
            all_result_dic[present_test_dic_key] = {}
            for key, index_list in test_dic[present_test_dic_key].items():
                all_result_dic[present_test_dic_key][key] = 1 if (len(
                    [element for element in present_result_set if element in index_list])) / len(
                    index_list) > 0.5 else 0

    for key, pos_res in all_result_dic.items():
        patient_res = 0
        for _, res in pos_res.items():
            if res:
                patient_res = 1
                break
        all_result_dic[key]['result'] = patient_res
        if key in present_id:
            all_result_dic[key]['target'] = 1
        else:
            all_result_dic[key]['target'] = 0
    result = []
    target = []
    for key, value in all_result_dic.items():
        result.append(value['result'])
        target.append(value['target'])
    # print('done')
    return result, target


def get_patient_res(res_idc: dict):
    """得到片段级和患者级准确率"""
    patient_res = {}
    for key, value in res_idc.items():
        patient_id = key.split('_')[0]
        if patient_id in patient_res.keys():
            patient_res[patient_id].append(value)
        else:
            patient_res[patient_id] = [value]
    for key, value in patient_res.items():
        patient_res[key] = np.mean(value)
    return patient_res


def segment_classifier(result_list_1: list, test_fold: list, set_path: str):
    """
    本fn计算了针对每个location和patient的acc和cm
    Args:
        result_list_1 (list, optional): 此列表用来存储分类结果为1对应的id.从test结果中生成传入.
        segment_present (list, optional): _description_. 这是有杂音（=1）的音频target列表，在列表中对应为1，不在则对应为0.
    Returns:
        _type_: _description_
    """
    global absent_test_index, present_test_index, present_test_names, absent_test_names
    npy_path_padded = rf"{set_path}\npyFile_padded\npy_files01_norm"
    # if len(test_fold) == 1:
    for k in test_fold:
        absent_test_index = np.load(npy_path_padded + f"\\absent_index_norm01_fold{k}.npy", allow_pickle=True)
        present_test_index = np.load(npy_path_padded + f"\\present_index_norm01_fold{k}.npy", allow_pickle=True)
        absent_test_names = np.load(npy_path_padded + f"\\absent_name_norm01_fold{k}.npy", allow_pickle=True)
        present_test_names = np.load(npy_path_padded + f"\\present_name_norm01_fold{k}.npy", allow_pickle=True)
    # todo 可以测一下这个字典names,index组合对不对
    absent_test_dic = dict(zip(absent_test_names, absent_test_index))
    present_test_dic = dict(zip(present_test_names, present_test_index))
    # 所有测试数据的字典
    test_dic = {**absent_test_dic, **present_test_dic}
    # 创建id_pos:idx的字典
    id_idx_dic = {}
    # 遍历test_dic，生成id_pos:idx的字典
    for file_name, data_index in test_dic.items():
        id_pos = file_name.split('_')[0] + '_' + file_name.split('_')[1]
        # 如果id_pos不在字典中，就创建一个新的键值对
        if id_pos in id_idx_dic.keys():
            id_idx_dic[id_pos].append(data_index)
            # 如果id_pos在字典中，就把value添加到对应的键值对的值中
        else:
            id_idx_dic[id_pos] = [data_index]
    # id_idx_dic格式：12345_AV: [001,002,003,004,005]
    # 创建一个空字典，用来存储分类结果,formate: id_pos: result
    result_dic = {}
    # 这样就生成了每个听诊区对应的数据索引，然后就可以根据索引读取数据了
    for id_pos, data_index in id_idx_dic.items():
        # 创建空列表用于保存数据索引对应的值
        value_list = []
        # 遍历这个id_pos对应的所有数据索引
        for idx in data_index:
            # 根据索引读取数据
            if idx in result_list_1:
                value_list.append(1)
            else:
                value_list.append(0)
        # 计算平均值作为每一段的最终分类结果，大于0.5就是1，小于0.5就是0,返回字典
        result_dic[id_pos] = np.mean(value_list)
    # result_dic格式：12345_AV: 0.5, 12345_MV: 0.3
    # 获取segment_target_list,这是csv里面读取的有杂音的音频的id和位置
    segment_present, patient_dic, absent_test_id_all, present_test_id_all = get_segment_target_list(test_fold, set_type)
    # 创建两个列表，分别保存outcome和target列表
    segment_output = []
    segment_target = []
    # 最后，根据target_list，将分类结果转换为0和1并产生outcome_list
    for id_loc, result_value in result_dic.items():
        # TODO: 这里的阈值是0.4,会提升性能吗？
        if result_value >= 0.5:
            segment_output.append(1)
            result_dic[id_loc] = 1
        else:
            segment_output.append(0)
            result_dic[id_loc] = 0
        # 这里计算segment的target
        if id_loc in segment_present:
            segment_target.append(1)
        else:
            segment_target.append(0)
    # 这里需要检查一下segment_target_list和segment_output_list的长度是否一致
    # 计算准确率的代码是这样的吗？？？
    segment_acc = (np.array(segment_output) == np.array(segment_target)).sum() / len(segment_target)
    # 计算混淆矩阵
    segment_cm = confusion_matrix(segment_target, segment_output)
    # -------------------------------------------------------- #
    # -----------------/ patient classifier /----------------- #
    # -------------------------------------------------------- #
    # patient_result_dic用于保存每个患者每个听诊区的分类结果，formate: id: location1_result,location2_result
    patient_result_dic = {}
    # patient_dic，formate: id:听诊区，123：AV+MV， 456：PV+TV
    for patient_id, locations in patient_dic.items():
        locations = locations.split('+')
        for location in np.unique(locations):
            # 这里除去了phc，因为phc不在result_dic中
            if location in ['AV', 'PV', 'TV', 'MV']:
                id_location = patient_id + '_' + location
            if id_location in result_dic.keys():
                if not patient_id in patient_result_dic.keys():
                    patient_result_dic[patient_id] = result_dic[id_location]
                else:
                    patient_result_dic[patient_id] += result_dic[id_location]
            else:
                # patient_result_dic[patient_id] += 0
                # 正常情况不会报这个错，因为result_dic中的id_loc都是在segment_target_list中的
                print('[WANGING 3]: ' + id_location + ' not in result_dic')
    # 遍历patient_result_dic，计算每个患者的最终分类结果
    patient_output_dic = {}
    patient_output = []
    patient_target = []
    for patient_id, result in patient_result_dic.items():
        # 做output
        if result == 0:  # 全为0 absent
            patient_output_dic[patient_id] = 0
            patient_output.append(0)
        elif result > 0:  # 不全为0 present
            patient_output_dic[patient_id] = 1
            patient_output.append(1)
        else:
            print('[WANGING 4]: result value is negtive?')  # 有负数
        # 做target
        if patient_id in absent_test_id_all:
            patient_target.append(0)
        elif patient_id in present_test_id_all:
            patient_target.append(1)
        else:
            print('[WANGING 5]: ' + patient_id + ' not in test_id')
    # 统计patient的错误id
    patient_id_test = list(patient_output_dic.keys())
    # 保存错误的id
    patient_error_id = []
    for patient_index in range(len(patient_output)):
        if patient_output[patient_index] != patient_target[patient_index]:
            patient_error_id.append(patient_id_test[patient_index])
    print(patient_error_id)

    # # 计算准确率和混淆矩阵
    # test_input, test_target)
    # # 计算准确率
    # #  patient_output.eq(patient_target).sum().item()
    # patient_acc = (np.array(patient_output) == np.array(
    #     patient_target)).sum()/len(patient_target)
    # # 计算混淆矩阵
    # patient_cm = confusion_matrix(
    #     patient_target, patient_output)
    return segment_acc, segment_cm, patient_output, patient_target, patient_error_id


def get_segment_target_list(test_fold, set_type):
    """ get segment target list
        根据csv文件生成并返回segment_target_list
        列表包含所有present的id和对应的位置
    """
    # if len(test_fold) == 1:
    for k in test_fold:
        absent_test_id_path = fr"D:\Shilong\new_murmur\02_dataset" + set_type + fr"\absent_fold_{k}.csv"
        present_test_id_path = fr"D:\Shilong\new_murmur\02_dataset" + set_type + fr"\present_fold_{k}.csv"
        absent_test_id = csv_reader_cl(absent_test_id_path, 0)
        present_test_id = csv_reader_cl(present_test_id_path, 0)
    # else:
    #     for i in range(len(test_fold)):
    #         k = test_fold[i]
    #         absent_test_id_path = fr"D:\Shilong\murmur\01_dataset" + \
    #             set_type+fr"\absent_fold_{k}.csv"
    #         present_test_id_path = fr"D:\Shilong\murmur\01_dataset" + \
    #             set_type+fr"\present_fold_{k}.csv"
    #         absent_test_id = csv_reader_cl(absent_test_id_path, 0)
    #         present_test_id = csv_reader_cl(present_test_id_path, 0)
    #         if i == 0:
    #             absent_test_id_all = absent_test_id
    #             present_test_id_all = present_test_id
    #         else:
    #             absent_test_id_all = np.concatenate(
    #                 (absent_test_id_all, absent_test_id), axis=0)
    #             present_test_id_all = np.concatenate(
    #                 (present_test_id_all, present_test_id), axis=0)
    csv_path = r"D:\Shilong\murmur\dataset_all\training_data.csv"
    # get dataset tag from table
    row_line = csv_reader_row(csv_path, 0)
    tag_list = []
    # get index for 'Patient ID' and 'Outcome'
    tag_list.append(row_line.index("Patient ID"))
    tag_list.append(row_line.index("Murmur"))
    tag_list.append(row_line.index("Murmur locations"))
    tag_list.append(row_line.index("Recording locations:"))
    id_data = csv_reader_cl(csv_path, tag_list[0])
    Murmur = csv_reader_cl(csv_path, tag_list[1])
    Murmur_locations = csv_reader_cl(csv_path, tag_list[2])
    Recording_locations = csv_reader_cl(csv_path, tag_list[3])
    # 测试集中所有的id
    test_id = absent_test_id + present_test_id
    # 创建一个空列表segment_present，用来存储有杂音的音频的id和位置
    segment_present = []
    # print(absent_test_id)
    for id in present_test_id:
        # 查此id的murmur状态和听诊位置
        murmurs = Murmur[id_data.index(id)]
        murmur_locations = Murmur_locations[id_data.index(id)]
        if murmurs == 'Present':
            locations = murmur_locations.split('+')
            for loc in locations:
                if loc in ['AV', 'PV', 'TV', 'MV']:
                    # 以id_loc的形式存储present的id和位置
                    segment_present.append(id + '_' + loc)
                else:
                    print(f'[WANGING 1]: {id}_{loc} not in locations')
        else:
            print(f'[WANGING 2]: {id} murmurs is not present?')

    patient_dic = {}
    present_id_loc = {}
    # 创建一个空字典，用来存储id和对应的听诊区,formate: id:听诊区
    # present_id_loc用于保存有杂音的id和对应的听诊区：id：AV+PV+TV
    # print(absent_test_id)
    for id in test_id:
        patient_dic[id] = Recording_locations[id_data.index(id)]
        present_id_loc[id] = Murmur_locations[id_data.index(id)]
    # , present_id_loc
    return segment_present, patient_dic, absent_test_id, present_test_id
