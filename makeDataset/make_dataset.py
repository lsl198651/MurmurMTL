# Author: Shi long
# Create Data: 2023-6
# Modify Date:2024-10-21

from utils.util_dataset import *

# ==================================================================== #
# ========================/ code executive /========================== #
# ==================================================================== #
if __name__ == '__main__':
    csv_path = r"E:\Shilong\00_PCGDataset\training_data.csv"
    # 分折的csv文件【修改】
    csv_source_path = r"D:\Shilong\new_murmur\02_dataset\02_s1s2_4k"
    # TODO 修改此处的root_path
    root_path = r"E:\Shilong\02_dataset\02_5s_4k_txt"
    wav_len = 5  # 按照wav_len s切割
    is_states = False  # 是否按照state切割据集
    random_flod = True  # 是否随机分折
    # TODO 修改此处的src_path
    src_path = r"E:\Shilong\00_PCGDataset\training_data"
    data_set(root_path, is_states, wav_len)

    # get dataset tag from table
    row_line = csv_reader_row(csv_path, 0)
    tag_list = [row_line.index("Patient ID"), row_line.index("Murmur"), row_line.index("Murmur locations"),
                row_line.index("Systolic murmur timing"), row_line.index("Diastolic murmur timing")]

    # for tag_index in tag_list:
    id_data = csv_reader_cl(csv_path, tag_list[0])
    Murmur = csv_reader_cl(csv_path, tag_list[1])
    Murmur_locations = csv_reader_cl(csv_path, tag_list[2])
    Systolic_murmur_timing = csv_reader_cl(csv_path, tag_list[3])
    Diastolic_murmur_timing = csv_reader_cl(csv_path, tag_list[4])

    mkdir(root_path)
    # save data to csv file
    pd.DataFrame(Murmur_locations).to_csv(root_path + r"\Murmur_locations.csv", index=False, header=False)
    pd.DataFrame(Systolic_murmur_timing).to_csv(root_path + r"\Systolic_murmur_timing.csv", index=False, header=False)
    pd.DataFrame(Diastolic_murmur_timing).to_csv(root_path + r"\Diastolic_murmur_timing.csv", index=False, header=False)
    # init patient id list for absent present and unknown

    # get 'Absent' and 'Present' and 'Unknown' index
    absent_id = [out for out, Murmur in enumerate(Murmur) if Murmur == "Absent"]
    present_id = [out for out, Murmur in enumerate(Murmur) if Murmur == "Present"]
    absent_patient_id = []
    present_patient_id = []
    # get 'Absent' and 'Present' and 'Unknown' patients ID
    for aid in absent_id:
        absent_patient_id.append(id_data[aid])
    for pid in present_id:
        present_patient_id.append(id_data[pid])

    # save patient id as csv
    pd.DataFrame(data=absent_patient_id, index=None).to_csv(root_path + r"\absent_id.csv", index=False, header=False)
    pd.DataFrame(data=present_patient_id, index=None).to_csv(root_path + r"\patient_id.csv", index=False, header=False)
    if random_flod:
        # 对Present和Absent分五折
        fold_absent = fold_divide(absent_patient_id, fold_num=5)
        fold_present = fold_divide(present_patient_id, fold_num=5)
        # 对Present和Absent分五折
        # 分别保存每折的id
        for k, v in fold_absent.items():
            pd.DataFrame(data=v, index=None).to_csv(root_path + r"\absent_fold_" + str(k) + ".csv", index=False,
                                                    header=False)
        for k, v in fold_present.items():
            pd.DataFrame(data=v, index=None).to_csv(root_path + r"\present_fold_" + str(k) + ".csv", index=False,
                                                    header=False)
    else:
        for k in range(5):
            shutil.copy(csv_source_path + r"\absent_fold_" + str(k) + ".csv", root_path + "\\")
            shutil.copy(csv_source_path + r"\present_fold_" + str(k) + ".csv", root_path + "\\")
        fold_absent = read_fold_csv(root_path, "absent")
        fold_present = read_fold_csv(root_path, "present")

    # define path options
    position = ["_AV", "_MV", "_PV", "_TV"]
    murmur_class = ["Absent", "Present"]
    period = ["s1", "systolic", "s2", "diastolic"]

    folder_path = root_path + "\\"

    # 将wav文件和tsv文件copy到目标文件夹
    copy_wav_file(src_path, folder_path, absent_patient_id, "Absent", position)
    copy_wav_file(src_path, folder_path, present_patient_id, "Present", position)

    # 创建每个wav文件的文件夹
    for mur in murmur_class:
        dir_path = folder_path + mur + "\\"
        for patient_id in absent_patient_id:
            pos_dir_make(dir_path, patient_id, position)
        for patient_id in present_patient_id:
            pos_dir_make(dir_path, patient_id, position)

    # 切数据，命名格式为：id+pos+state+num
    # absent
    period_div(folder_path, murmur_class[0] + "\\", absent_patient_id, position, id_data, Murmur_locations,
               Systolic_murmur_timing, Diastolic_murmur_timing, segments_len=wav_len, is_state_cut=is_states)
    # present
    period_div(folder_path, murmur_class[1] + "\\", present_patient_id, position, id_data, Murmur_locations,
               Systolic_murmur_timing, Diastolic_murmur_timing, segments_len=wav_len, is_state_cut=is_states)

    absent_train_id_path = root_path + r"\absent_train_id.csv"
    absent_test_id_path = root_path + r"\absent_test_id.csv"
    present_train_id_path = root_path + r"\present_train_id.csv"
    present_test_id_path = root_path + r"\present_test_id.csv"

    # 按照每折的id复制数据到每折对应文件夹
    # 此处执行后的数据，数据只按折分开了，并没有按present和Absnet分开
    for k, v in fold_absent.items():
        copy_states_data(v, root_path, "\\fold_" + str(k), "\\Absent\\")
    for k, v in fold_present.items():
        copy_states_data(v, root_path, "\\fold_" + str(k), "\\Present\\")

    for k in range(5):
        for murmur_class in ['Absent', 'Present']:
            src_fold_path = root_path + r"\fold_" + str(k) + "\\" + murmur_class + "\\"
            target_dir = root_path + r'\fold_set_' + str(k)
            mkdir(target_dir)
            mkdir(target_dir + "\\absent\\")
            mkdir(target_dir + "\\present\\")
            for root, dir, file in os.walk(src_fold_path):
                for subfile in file:
                    files = os.path.join(root, subfile)
                    print(subfile)
                    state = subfile.split("_")[4]
                    if state == 'Absent':
                        shutil.copy(files, target_dir + "\\absent\\")
                    else:
                        shutil.copy(files, target_dir + "\\present\\")
                    # else:
                    #     raise ValueError("state error")

    for k in range(5):
        src_fold_root_path = root_path + r"'\fold_set_" + str(k)
        for murmur_class in ['absent', 'present']:
            src_fold_path = src_fold_root_path + "\\" + murmur_class + "\\"

    data_set(root_path, is_states, wav_len)
