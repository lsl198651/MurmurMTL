import csv
import random
import shutil

import librosa
import librosa.display
import numpy as np
import pandas as pd
import soundfile

from utils.helper_code import *
from utils.util_features import get_features_mod, get_logmel_feature
from utils.util_saveInfo import *


def mkdir(path):
    # judge weather make dir or not
    if not os.path.exists(path):
        os.makedirs(path)


def csv_reader_cl(file_name, clo_num):
    """
    按列读文件，列号从0开始
    """
    with open(file_name, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[clo_num] for row in reader]
    return column


#
def csv_reader_row(file_name, row_num):
    """read the csv row_num-th row"""
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        row = list(reader)
    return row[row_num]


def copy_file(src_path, folder_path, patient_id_list, mur, position):
    """将所有文件复制到目标目录"""
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir = folder_path + "\\" + mur + "\\" + patient_id + "\\"
            os.makedirs(target_dir, exist_ok=True)

            txt_name = src_path + "\\" + patient_id + ".txt"
            wav_name = src_path + "\\" + patient_id + pos + ".wav"
            hea_name = src_path + "\\" + patient_id + pos + ".hea"
            tsv_name = src_path + "\\" + patient_id + pos + ".tsv"
            if os.path.exists(txt_name):
                shutil.copy(txt_name, target_dir + "\\")
            if os.path.exists(wav_name):
                shutil.copy(wav_name, target_dir + "\\")
            if os.path.exists(hea_name):
                shutil.copy(hea_name, target_dir + "\\")
            if os.path.exists(tsv_name):
                shutil.copy(tsv_name, target_dir + "\\")


def copy_wav_file(src_path, folder_path, patient_id_list, mur, position):
    """将wav和tsv文件复制到目标目录"""
    count = 0
    # 1. make dir
    mur_dir = folder_path + "\\" + mur
    mkdir(mur_dir)
    # 2. copy file
    for patient_id in patient_id_list:
        # for mur in murmur:
        for pos in position:
            target_dir = folder_path + "\\" + mur + "\\" + patient_id + "\\"
            os.makedirs(target_dir, exist_ok=True)
            wav_name = src_path + "\\" + patient_id + pos + ".wav"
            tsv_name = src_path + "\\" + patient_id + pos + ".tsv"
            txt_name = src_path + "\\" + patient_id + ".txt"
            if os.path.exists(wav_name):
                shutil.copy(wav_name, target_dir + "\\")
                count += 1
            if os.path.exists(tsv_name):
                shutil.copy(tsv_name, target_dir + "\\")
            if os.path.exists(txt_name):
                shutil.copy(txt_name, target_dir + "\\")
    print("copy file num: ", count)


# devide sounds into 4s segments
def pos_dir_make(dir_path, patient_id, pos):
    for po in pos:
        subdir = dir_path + patient_id + "\\" + patient_id + po
        wav_name = subdir + ".wav"
        if os.path.exists(wav_name):
            print("exist")
            mkdir(subdir)  # make dir


def index_load(tsvname):
    """读取tsv文件内容,不需要close函数"""
    with open(tsvname, "r") as f:
        txt_data = f.read()
    head = ["start", "end", "period"]
    data = txt_data.split("\n")[:-1]
    # 遍历每一行
    for l in data:
        sgmt = l.split("\t")
        if sgmt[2] != "0":
            head = np.vstack([head, sgmt])
    return head[1:]


def period_div(
        path,
        mur,
        patient_id_list,
        position,
        id_data,
        murmur_locations,
        systolic_murmur_timing,
        diastolic_murmur_timing,
        segments_len,
        is_state_cut=False

):
    """
    按照时相切割
    """
    for patient_id in patient_id_list:
        patient_dir_path = path + mur + patient_id + "\\" + patient_id
        txt_path = patient_dir_path + ".txt"
        current_patient_data = load_patient_data(txt_path)
        human_feat = get_features_mod(current_patient_data)
        for pos in position:
            dir_path = path + mur + patient_id + "\\" + patient_id + pos
            tsv_path = dir_path + ".tsv"
            wav_path = dir_path + ".wav"
            index = id_data.index(patient_id)
            wav_location = pos[1:]  # 听诊区域
            locations = murmur_locations[index].split("+")  # 有杂音的区域
            # 此听诊区有杂音
            if wav_location in locations:
                murmur_type = "Present"
                systolic_state = systolic_murmur_timing[index]
                diastolic_state = diastolic_murmur_timing[index]
                # 没有 Systolic murmur
                systolic_murmur = "Absent" if systolic_state == "nan" else "Present"
                # 没有 Diastolic murmur
                diastolic_murmur = "Absent" if diastolic_state == "nan" else "Present"
                murmur_type = f'{systolic_state}+{diastolic_state}'
            # 此听诊区没有杂音
            else:
                murmur_type, systolic_murmur, diastolic_murmur, systolic_state, diastolic_state = "Absent", "Absent", "Absent", "nan", "nan"

            if os.path.exists(tsv_path):
                if is_state_cut:
                    state_div(tsv_path, wav_path, dir_path + "\\", patient_id + pos, systolic_murmur, diastolic_murmur,
                              systolic_state, diastolic_state, human_feat)
                else:
                    duration_div(tsv_path, wav_path, dir_path + "\\", patient_id + pos, murmur_type, segments_len,
                                 human_feat=human_feat)


def state_div(
        tsvName,
        wavName,
        statePath,
        id_pos,
        SystolicMurmur,
        diastolicMurmur,
        systolic_state,
        diastolic_state,
        human_feat
):
    """
    按照识相切割
    """
    index_file = index_load(tsvName)
    recording, fs = librosa.load(wavName, sr=4000)
    num = 0

    for i in range(index_file.shape[0] - 3):
        if index_file[i][2] == "1" and index_file[i + 3][2] == "4":
            startIndex1 = float(index_file[i][0]) * fs
            endIndex1 = float(index_file[i + 1][1]) * fs
            startIndex2 = float(index_file[i + 2][0]) * fs
            endIndex2 = float(index_file[i + 3][1]) * fs
            num = num + 1
            print("==========state_div===========")
            buff1 = recording[int(startIndex1): int(endIndex1)]  # 字符串索引切割
            buff2 = recording[int(startIndex2): int(endIndex2)]  # 字符串索引切割
            print("buff1 len: " + str(len(buff1)), "buff2 len: " + str(len(buff2)))
            soundfile.write(
                statePath
                + f"{id_pos}_s1+Systolic_{num}_{SystolicMurmur}_{systolic_state}_{human_feat}.wav",
                buff1,
                fs
            )
            # 切舒张期
            soundfile.write(
                statePath
                + f"{id_pos}_s2+Diastolic_{num}_{diastolicMurmur}_{diastolic_state}_{human_feat}.wav",
                buff2,
                fs,

            )


def duration_div(
        tsv_name,
        wav_name,
        state_path,
        id_pos,
        murmur_type,
        duration_len,
        human_feat
):
    """
    按照固定时长切片
    切割长度为spilt_len s
    """
    label_frame = 20  # 每label_frame 毫秒一个label
    index_file = index_load(tsv_name)
    recording, fs = librosa.load(wav_name, sr=4000)
    labels = np.full(len(recording) // 80, 0)
    print(f'lableslen:{len(labels)},recordings_len:{len(recording)}')
    for (start, end, tag) in index_file:
        labels[int(float(start) * 1000 // label_frame):int(float(end) * 1000 // label_frame)] = int(tag)

    # print(labels)

    # 从0开始切割
    start = float(index_file[0][0]) * fs
    end = float(index_file[-1][1]) * fs
    recording_buff = recording[int(start): int(end)]  # 准备切割的数据
    labels_buff = labels[int(float(index_file[0][0]) * 1000 // label_frame): int(
        float(index_file[-1][1]) * 1000 // label_frame)]  # 准备切割的标记
    fs = int(fs)
    # 计算每个片段的样本数
    samples_per_recording = duration_len * fs
    # 计算每个片段的标签长度
    points_per_tag = int(samples_per_recording // 80)
    # 切割音频数据
    segments = []

    for wav_start in np.arange(0, len(recording_buff), samples_per_recording):
        wav_end = wav_start + samples_per_recording
        segment = recording_buff[wav_start:wav_end]
        segments.append(segment)

    # 切割标签数据
    tags = []
    for tag_start in np.arange(0, len(labels_buff), points_per_tag):
        tag_end = tag_start + points_per_tag
        new_tags = labels_buff[tag_start:tag_end]
        tags.append(new_tags)

    if len(segments) > len(tags):
        segments.pop()

    # segments.pop()
    for num, segment in enumerate(segments):
        print("==========duration_div===========")
        wav_segment = segment
        tag_segment = tags[num]
        print("wav_segment len: " + str(len(wav_segment)))
        if len(wav_segment) < samples_per_recording:
            repeat_times = samples_per_recording // len(wav_segment)  # 重复次数，不用减一
            for i in range(repeat_times):
                wav_segment = np.hstack((wav_segment, segment))
                tag_segment = np.hstack((tag_segment, tags[num]))
            if len(wav_segment) < samples_per_recording:  # 应该不可能进这个，
                wav_segment = np.pad(
                    wav_segment,
                    (0, samples_per_recording - len(wav_segment)),
                    "constant",
                    constant_values=(0, 0)
                )
            else:
                wav_segment = wav_segment[:samples_per_recording]

            if len(tag_segment) < points_per_tag:
                tag_segment = np.pad(
                    tag_segment,
                    (0, points_per_tag - len(tag_segment)),
                    "constant",
                    constant_values=(0, 0)
                )
            else:
                tag_segment = tag_segment[:points_per_tag]
        if len(wav_segment) != samples_per_recording or len(tag_segment) != points_per_tag:
            raise ValueError("wav_segment and tag_segment length not equal")
            print(state_path)

        print(f"wav_segment len: {str(len(wav_segment))}, tag len: {len(tag_segment)}")

        file_path = state_path + "{}_{}_{}_{}_{}_{}".format(id_pos, str(duration_len) + "s", num, murmur_type,
                                                            "None", human_feat)
        soundfile.write(rf"{file_path}.wav", wav_segment, fs)
        save_as_txt_np(tag_segment, rf"{file_path}.txt")


# get patient id from csv file
def get_patientid(csv_path):
    # 'import csv' is required
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        id = [row["0"] for row in reader]  # weight 同列的数据
        return id


def fold_divide(data, fold_num=5):
    """五折交叉验证
    将输入列表打乱，然后分成五份
    output: flod5 = {0:[],1:[],2:[],3:[],4:[]}
    """
    # 打乱序列
    random.shuffle(data)
    # 五折
    flod5 = {}
    point = []
    for i in range(fold_num):
        point.append(i * round(len(data) / fold_num))
    # print(point)
    # 分割序列
    for i in range(len(point)):
        if i < len(point) - 1:
            flod5[i] = []
            flod5[i].extend(data[point[i]:point[i + 1]])
        else:
            flod5[i] = []
            flod5[i].extend(data[point[-1]:])
    return flod5


def read_fold_csv(root_path, murmur_type):
    fold = {}
    for k in range(5):
        csv_path = root_path + rf"\{murmur_type}_fold_" + str(k) + ".csv"
        # 读取每个折的文件并存储id
        fold[k] = []
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                # 假设每行只有一个id
                fold[k].append(row[0])
    return fold


# copy data to folder
def copy_states_data(patient_id, folder, type, murmur):
    traget_path = folder + type + murmur
    if not os.path.exists(traget_path):
        os.makedirs(traget_path)
    for id in patient_id:
        dir_path = folder + murmur + id
        print(dir_path)
        for root, dir, file in os.walk(dir_path):
            for subdir in dir:
                subdir_path = os.path.join(root, subdir)
                print(subdir_path)
                if os.path.exists(dir_path):
                    shutil.copytree(subdir_path, traget_path + subdir)
                else:
                    print("dir not exist")


def data_set(root_path, is_by_state, wav_len):
    """数据增强，包括时间拉伸和反转"""
    global data_id
    npy_path_padded = root_path + r"\npyFile_padded\npy_files01_norm"
    index_path = root_path + r"\npyFile_padded\index_files01_norm"
    mkdir(npy_path_padded)
    mkdir(index_path)
    for k in range(5):

        src_fold_root_path = root_path + r"\fold_set_" + str(k)
        # TODO 是否做数据增强
        # data_Auge(src_fold_root_path)
        for folder in os.listdir(src_fold_root_path):
            dataset_path = os.path.join(src_fold_root_path, folder)
            if k == 0 and folder == "absent":
                wav, label, names, index, data_id, feat, tags = get_wav_data(dataset_path, is_by_state, wav_len,
                                                                             data_id=0)  # absent
            else:
                wav, label, names, index, data_id, feat, tags = get_wav_data(dataset_path, is_by_state, wav_len,
                                                                             data_id)  # absent
            mel_list = []
            for i in range(len(wav)):
                mel = get_logmel_feature(wav[i])
                mel_list.append(mel)
            np.save(npy_path_padded + f"\\{folder}_mel_norm01_fold{k}.npy", mel_list)  # mel谱特征
            np.save(npy_path_padded + f"\\{folder}_wav_norm01_fold{k}.npy", wav)  # 原始数据
            np.save(npy_path_padded + f"\\{folder}_labels_norm01_fold{k}.npy", label)  # 标签
            np.save(npy_path_padded + f"\\{folder}_index_norm01_fold{k}.npy", index)  # 索引
            np.save(npy_path_padded + f"\\{folder}_name_norm01_fold{k}.npy", names)  # 文件名
            np.save(npy_path_padded + f"\\{folder}_feat_norm01_fold{k}.npy", feat)  # 人口特征
            np.save(npy_path_padded + f"\\{folder}_tags_norm01_fold{k}.npy", tags)  # 状态标签
            absent_train_dic = zip(index, names, feat)
            pd.DataFrame(absent_train_dic).to_csv(index_path + f"\\fold{k}_{folder}_disc.csv", index=False,
                                                  header=False)

    output_path = root_path + r"\npyFile_padded\organized_data"
    mkdir(output_path)
    data_path = root_path + r"\npyFile_padded\index_files01_norm"
    for root, dir, file in os.walk(data_path):
        for file in file:
            if file.endswith(".csv"):
                print('processing file:', file)
                get_id_position_org(root, output_path, file)

    print("data set is done!")


def get_wav_data(dir_path, is_by_state, time, data_id=0):
    """返回数据文件"""
    waves = []
    labels = []
    names = []
    index = []
    feats = []
    # tags = []
    # 设置采样率为4k，时间长度为4
    tags=np.zeros(250)
    fs = 4000

    if is_by_state:
        data_length = 1300
    else:
        data_length = fs * time
    for root, dir, file in os.walk(dir_path):
        for subfile in file:
            if subfile.endswith(".wav"):
                file_name = os.path.splitext(subfile)[0]
                wav_name = file_name + ".wav"
                txt_name = file_name + ".txt"
                # if wav_name.endswith(".wav"):
                # 报错

                wav_path = os.path.join(root, wav_name)
                if os.path.exists(wav_path) :
                    # 序号
                    data_id = data_id + 1
                    names.append(subfile)
                    index.append(data_id)
                    # 数据读取
                    print("reading: " + subfile)
                    y, sr = librosa.load(wav_path, sr=4000)
                    # TODO 采样率:4k
                    y_4k_norm = wav_normalize(y)  # 归一化
                    # # 数据裁剪
                    if y_4k_norm.shape[0] < data_length:
                        y_4k_norm = np.pad(
                            y_4k_norm,
                            ((0, data_length - y_4k_norm.shape[0])),
                            "constant",
                            constant_values=(0, 0)
                        )
                    elif y_4k_norm.shape[0] > data_length:
                        # y_4k_norm = y_4k_norm[-data_length:]
                        y_4k_norm = y_4k_norm[:data_length]
                    print("index is " + str(data_id), "y_4k size: " + str(y_4k_norm.size))

                    waves.append(y_4k_norm)
                    file_name = subfile.split("_")
                    # 标签读取
                    if file_name[4] == "Absent":  # Absent
                        labels.append(0)
                    elif file_name[4] == "Present":  # Present
                        labels.append(1)  # 说明该听诊区有杂音
                    feats.append(file_name[-1])

                    tags_seg = read_txt_np(os.path.join(root, txt_name))
                    tags=np.vstack((tags,tags_seg))

    # 将列表转换为NumPy数组
    # waves = np.array(waves)
    # labels = np.array(labels)
    # index = np.array(index)
    # tags=np.array(tags)

    return waves, labels, names, index, data_id, feats, tags[1:]

def wav_normalize(data):
    """min max归一化"""
    # range = np.max(data) - np.min(data)
    data = (data - np.mean(data)) / np.max(np.abs(data))
    # data = (data-np.min(data))/range
    return data
    # recording -= recording.mean()
    # recording /= recording.abs().max()


def extract_id_and_auscultation_area(filename):
    # Split by non-digit characters to extract the ID and auscultation area
    parts = filename.split('_')
    if len(parts) > 1:
        id_part = ''.join(filter(str.isdigit, parts[0]))  # Extract digits from the first part
        auscultation_area_part = parts[1].split('+')[0]  # Extract the part before '+' sign
        return int(id_part) if id_part.isdigit() else None, auscultation_area_part
    else:
        return None, None


def get_id_position_org(file_path, output_file_path, file_name):
    # Load the CSV file without headers
    csv_name = file_path + "\\" + file_name
    data = pd.read_csv(csv_name, header=None)

    # Apply the adjusted function to extract ID and auscultation area
    data[['id', 'auscultation_area']] = data.iloc[:, 1].apply(
        lambda x: pd.Series(extract_id_and_auscultation_area(x)))

    # Group by 'ID' and 'Auscultation_Area' and aggregate the indices from the first column
    grouped = data.groupby(['id', 'auscultation_area']).agg({0: list}).reset_index()

    # Rename columns for clarity
    grouped.columns = ['id', 'auscultation_area', 'indices']

    # Save the grouped data to a new CSV file with headers
    output_file = output_file_path + r'\organized_data_' + file_name
    grouped.to_csv(output_file, index=False)


# 将CSV数据转换为更结构化的字典格式
# 格式为：{ID: {Auscultation_Area: [Indices]}}
# 定义一个函数来读取CSV文件并将其转换为所需的字典格式
def csv_to_dict(csv_filepath):
    structured_dict = {}

    # 打开CSV文件并读取
    with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id_key = row['id']
            auscultation_area = row["auscultation_area"]
            indices_str = row['indices']

            # 将字符串格式的索引列表转换为实际的列表
            indices_list = [int(i.strip()) for i in indices_str.strip('[]').split(', ')]

            # 如果ID不在结构化字典中，则添加它
            if id_key not in structured_dict:
                structured_dict[id_key] = {}

            # 将听诊区和对应的索引列表添加到结构化字典中
            structured_dict[id_key][auscultation_area] = indices_list

    return structured_dict
