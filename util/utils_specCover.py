import pandas as pd

from helper_code import *


def cover2embedings(data_folder, csv_path):
    """
        生成一个patient-embedding的csv文件
    """

    patient_id = ['Patient ID']
    locations = ['Recording locations:']
    pregnancy_status = ['Pregnancy status']

    patient_files = find_patient_files(data_folder)
    total_spec = ["embedding"]
    for i in range(len(patient_files)):
        current_patient_data = load_patient_data(patient_files[i])
        pid = get_patient_id(current_patient_data)
        location = get_locations(current_patient_data)

        a = get_age(current_patient_data)
        s = get_sex(current_patient_data)
        h = get_height(current_patient_data)
        w = get_weight(current_patient_data)
        ps = get_pregnancy_status(current_patient_data)

        pregnancy_status.append(ps)
        patient_id.append(pid)
        locations.append(location)
        total_spec.append([a, s, h, w])

    info = zip(pid, location, total_spec)
    pd.DataFrame(info).to_csv(f"{csv_path}.csv", index=False, header=False)


if __name__ == '__main__':
    data_folder = r'D:\Shilong\murmur\Dataset\PCGdataset\test_data'
    cover2embedings(data_folder, csv_path='test_data')
