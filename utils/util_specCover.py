import pandas as pd

from helper_code import *


def set2list(set_data):
    return list(set_data)


class SpecCover:
    def __init__(self, data_folder):
        self.a_list = []
        self.s_list = []
        self.h_list = []
        self.w_list = []
        self.ps_list = []
        self.data_folder = data_folder
        self.patient_files = find_patient_files(data_folder)
        self.patient_data = [load_patient_data(patient_file) for patient_file in self.patient_files]

        self.a_set = set()
        self.s_set = set()
        self.h_set = set()
        self.w_set = set()
        self.ps_set = set()
        self.patient_id = ["patient_id"]
        self.locations = ['Recording locations:']
        self.embeddings = ['embedding']
        self.total_spec = ["embedding"]

    #     将集合中的元素去重，每个元素对应一个索引
    def emb_list(self):
        """
            生成一个patient-embedding的csv文件
        """

        patient_files = find_patient_files(self.data_folder)

        for i in range(len(patient_files)):
            current_patient_data = load_patient_data(patient_files[i])
            self.a_set.add(get_age(current_patient_data))
            self.s_set.add(get_sex(current_patient_data))
            self.h_set.add(get_height(current_patient_data))
            self.w_set.add(get_weight(current_patient_data))
            self.ps_set.add(get_pregnancy_status(current_patient_data))
            # pregnancy_status.append(ps)
        self.a_list = set2list(self.a_set)
        self.s_list = set2list(self.s_set)
        self.h_list = set2list(self.h_set)
        self.w_list = set2list(self.w_set)
        self.ps_list = set2list(self.ps_set)

    def cover2embedings(self, csv_path):
        """
            生成一个patient-embedding的csv文件
        """
        patient_files = find_patient_files(self.data_folder)

        for i in range(len(patient_files)):
            current_patient_data = load_patient_data(patient_files[i])
            a = get_age(current_patient_data)
            s = get_sex(current_patient_data)
            h = get_height(current_patient_data)
            w = get_weight(current_patient_data)
            ps = get_pregnancy_status(current_patient_data)
            self.patient_id.append(get_patient_id(current_patient_data))
            self.locations.append(get_locations(current_patient_data))

            self.a_set.add(get_age(current_patient_data))
            self.total_spec.append(
                [self.a_list.index(a), self.s_list.index(s), self.h_list.index(h), self.w_list.index(w),
                 self.ps_list.index(ps)]
            )

        info = zip(self.patient_id, self.locations, self.total_spec)
        pd.DataFrame(info).to_csv(f"{csv_path}.csv", index=False, header=False)


if __name__ == '__main__':
    folder = r'D:\Shilong\murmur\Dataset\PCGdataset\test_data'
    m = SpecCover(folder)
    m.emb_list()
    m.cover2embedings(csv_path='test_data')
