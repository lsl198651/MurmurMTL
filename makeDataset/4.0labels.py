import os
import csv
import numpy as np
import wave

# 路径设置
tsv_files_path = 'D:\project\LHH_MTL\MLT_data1234 _2\processdata/test'  # 替换为实际的TSV文件路径
audio_files_path = 'D:\project\LHH_MTL\MLT_data1234 _2\processdata/test'  # 替换为实际的音频文件路径
npsavefile_seg = 'D:\project\LHH_MTL\MLT_data1234 _2\labels/test_labels.npy'
npsavefile_noi = 'D:\project\LHH_MTL\MLT_data1234 _2\labels2/test_labels.npy'

time_len = 10

def replace_zeros(arr):
    # 将数组展平为一维，以便于遍历
    flattened_arr = arr.flatten()

    # 遍历数组从第二个元素开始
    for i in range(1, len(flattened_arr)):
        if flattened_arr[i] == 0:
            flattened_arr[i] = flattened_arr[i - 1]  # 替换为前一个非零值

    # 将展平的数组恢复原始形状
    return flattened_arr.reshape(arr.shape)

def get_half_indices(start, end, mode='first'):
    """ Helper function to get indices for marking segments """
    length = end - start
    half_length = length // 2
    if mode == 'first':
        return range(start, end)
    elif mode == 'middle':
        return range(start, end)
    elif mode == 'all':
        return range(start, end)
    elif mode == 'last':
        return range(start, end)
    elif mode == 'none':
        return []  # Return an empty range, so no indices will be set to 1

def get_half_indices_true(start, end, mode='first'):
    """ Helper function to get indices for marking segments """
    length = end - start
    half_length = length // 2
    if mode == 'first':
        return range(start, start + half_length)
    elif mode == 'middle':
        return range(start + (length - half_length) // 2, start + (length + half_length) // 2)
    elif mode == 'all':
        return range(start, end)
    elif mode == 'last':
        return range(start + half_length, end)
    elif mode == 'none':
        return []  # Return an empty range, so no indices will be set to 1

def construct_new_annotations(original_annotations, murmur_type):
    new_annotations = [0] * len(original_annotations)
    i = 0
    while i < len(original_annotations):
        start = i
        label = original_annotations[i]
        while i < len(original_annotations) and original_annotations[i] == label:
            i += 1
        end = i

        # Handle the annotations based on murmur type and label
        if murmur_type == 'Absent':
            continue
        elif murmur_type == 'Early-systolic-Early-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 2
        elif murmur_type == 'Early-systolic-Holodiastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 2
        elif murmur_type == 'Early-systolic-Mid-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 2
        elif murmur_type == 'Early-systolic-nan':
            if label == 2:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 1
        elif murmur_type == 'Mid-systolic-Early-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 2
        elif murmur_type == 'Mid-systolic-Holodiastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 2
        elif murmur_type == 'Mid-systolic-Mid-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 2
        elif murmur_type == 'Mid-systolic-nan':
            if label == 2:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 1
        elif murmur_type == 'Holosystolic-Early-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 2
        elif murmur_type == 'Holosystolic-Holodiastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 2
        elif murmur_type == 'Holosystolic-Mid-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 2
        elif murmur_type == 'Holosystolic-nan':
            if label == 2:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 1
        elif murmur_type == 'Late-systolic-Early-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'last'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 2
        elif murmur_type == 'Late-systolic-Holodiastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'last'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 2
        elif murmur_type == 'Late-systolic-Mid-diastolic':
            if label == 2:
                for j in get_half_indices(start, end, 'last'):
                    new_annotations[j] = 1
            elif label == 4:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 2
        elif murmur_type == 'Late-systolic-nan':
            if label == 2:
                for j in get_half_indices(start, end, 'last'):
                    new_annotations[j] = 1
        elif murmur_type == 'nan-Early-diastolic':
            if label == 4:
                for j in get_half_indices(start, end, 'first'):
                    new_annotations[j] = 2
        elif murmur_type == 'nan-Holodiastolic':
            if label == 4:
                for j in get_half_indices(start, end, 'all'):
                    new_annotations[j] = 2
        elif murmur_type == 'nan-Mid-diastolic':
            if label == 4:
                for j in get_half_indices(start, end, 'middle'):
                    new_annotations[j] = 2

        # Add more elif blocks for other combinations as needed

    return new_annotations

# 读取TSV文件并存储解析后的数据
def read_tsv_file(tsv_file):
    segments = []
    with open(tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            start_time = float(row[0])
            end_time = float(row[1])
            label = row[2]
            segments.append((start_time, end_time, int(label)))
    return segments


# 处理音频文件并标注
def label_audio_file(audio_file, segments):
    with wave.open(audio_file, 'rb') as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        duration = nframes / framerate
        audio_data = wf.readframes(nframes)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16)

    # 提取音频文件中的段号
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    segment_index = int(file_name.split('_')[-1])
    segment_start_time = segment_index * time_len  # 每段time_len0s

    # 按每20ms分段
    segment_duration = 0.02  # 20ms
    step_size = int(segment_duration * framerate)
    num_segments = int(np.ceil(duration / segment_duration))

    # 为每段音频标注
    labels = []

    for i in range(num_segments):
        segment_start = segment_start_time + i * segment_duration
        segment_end = segment_start_time + (i + 1) * segment_duration
        segment_mid = (segment_start + segment_end) / 2

        # 查找当前段落的标签
        current_label = 0  # 默认标签
        for (start_time, end_time, label) in segments:
            if start_time <= segment_mid < end_time:
                current_label = label
                break
            elif start_time < segment_end and end_time > segment_start:
                overlap_start = max(start_time, segment_start)
                overlap_end = min(end_time, segment_end)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > (segment_duration / 2):
                    current_label = label
                    break

        labels.append(current_label)

    return labels


# 主处理函数
def process_files(tsv_files_path, audio_files_path):
    all_labels = []   #seg
    all_labels2 = []  #noise
    tsv_files = sorted([f for f in os.listdir(tsv_files_path) if f.endswith('.tsv')])
    audio_files = sorted([f for f in os.listdir(audio_files_path) if f.endswith('.wav')])

    for tsv_file in tsv_files:
        file_prefix = os.path.splitext(tsv_file)[0]
        segments = read_tsv_file(os.path.join(tsv_files_path, tsv_file))

        for audio_file in audio_files:
            if audio_file.startswith(file_prefix):
                murmur_type = audio_file.split('_')[2]  # Assuming the format is Number_Position_MurmurType_Number.wav
                labels = label_audio_file(os.path.join(audio_files_path, audio_file), segments)
                labels_noi = construct_new_annotations(labels, murmur_type)
                all_labels.append(labels)
                all_labels2.append(labels_noi)
    # 将标签转换为矩阵
    max_length = max(len(labels) for labels in all_labels)
    labels_matrix = np.full((len(all_labels), max_length), 0)  # 用默认标签0填充
    for i, labels in enumerate(all_labels):
        labels_matrix[i, :len(labels)] = labels

    max_length2 = max(len(labels_noi) for labels_noi in all_labels2)
    labels_matrix2 = np.full((len(all_labels2), max_length2), 0)  # 用默认标签0填充
    for i, labels_noi in enumerate(all_labels2):
        labels_matrix2[i, :len(labels_noi)] = labels_noi

    return labels_matrix, labels_matrix2

# 执行处理
if __name__=="__main__":
    all_labels_seg, all_labels_noi = process_files(tsv_files_path, audio_files_path)
    all_labels_seg = replace_zeros(all_labels_seg)
    np.save(npsavefile_seg, all_labels_seg)
    np.save(npsavefile_noi, all_labels_noi)
