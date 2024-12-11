import os
from  util.helper_code import *
import shutil

if __name__=="__main__":
    files_path = r'E:\Shilong\00_PCGDataset\training_data'  # 替换为实际的TSV文件路径
    audio_files_path = r'E:\Shilong\00_PCGDataset\training_data'  # 替换为实际的音频文件路径
    new_files_path = r'E:\Shilong\00_PCGDataset\training_data\copy'
    audio_files = sorted([f for f in os.listdir(audio_files_path) if f.endswith('.wav')])
    tsv_files = sorted([f for f in os.listdir(files_path) if f.endswith('.txt')])

    patient_files = find_patient_files(files_path)
    for i in range(len(patient_files)):
        current_patient_data = load_patient_data(patient_files[i])
        patient_id = get_patient_id(current_patient_data)
        murmur_loca = get_murmur_locations(current_patient_data)
        systolic = get_systolic_murmur_timing(current_patient_data)
        diastolic = get_diastolic_murmur_timing(current_patient_data)

        for audio_file in audio_files:
            original_filename=os.path.join(files_path, audio_file)
            if audio_file.startswith(patient_id):
                wav_name=audio_file.split(".")[0]
                if wav_name.split('_')[-1] in  murmur_loca:
                    new_name = audio_file.replace(".wav", f"_{systolic}+{diastolic}.wav")
                else:
                    new_name=audio_file.replace(".wav", f"_Absent.wav")

                new_file_path = os.path.join(new_files_path, new_name)
                shutil.copy(original_filename, new_file_path)
                print(new_name)

