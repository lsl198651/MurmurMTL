import numpy as np


# args,


def get_features(set_path, train_fold: list, test_fold: list, set_name: str, train_folders):
    root_path = set_path + set_name
    npy_path_padded = root_path + r"\npyFile_padded\npy_files01_norm"
    train_feature_dic = {}
    train_wav_dic = {}
    train_labels_dic = {}
    train_index_dic = {}
    train_mel_dic = {}
    train_tags_dic = {}
    for k in train_fold:
        src_fold_root_path = root_path + r"\fold_set_" + k
        train_feature_dic[k] = {}
        train_wav_dic[k] = {}
        train_labels_dic[k] = {}
        train_index_dic[k] = {}
        train_mel_dic[k] = {}
        train_tags_dic[k] = {}
        # data_Auge(src_fold_root_path)
        # train_folders = ['absent', 'present', 'reverse0.8', 'reverse0.9', 'reverse1.0', 'reverse1.1',
        #                  'reverse1.2', 'time_stretch0.8', 'time_stretch0.9', 'time_stretch1.1', 'time_stretch1.2']

        for folder in train_folders:
            train_tags_dic[k][folder] = np.load(npy_path_padded +
                                                f"\\{folder}_tags_norm01_fold{k}.npy", allow_pickle=True)
            train_labels_dic[k][folder] = np.load(npy_path_padded +
                                                  f"\\{folder}_labels_norm01_fold{k}.npy", allow_pickle=True)
            train_index_dic[k][folder] = np.load(npy_path_padded +
                                                 f"\\{folder}_index_norm01_fold{k}.npy", allow_pickle=True)
            train_mel_dic[k][folder] = np.load(npy_path_padded +
                                               f"\\{folder}_mel_norm01_fold{k}.npy", allow_pickle=True)
    val_mel_dic = {}
    val_wav_dic = {}
    val_labels_dic = {}
    val_index_dic = {}
    val_ebd_dic = {}
    val_tags_dic = {}
    for v in test_fold:
        val_mel_dic[v] = {}
        val_wav_dic[v] = {}
        val_labels_dic[v] = {}
        val_index_dic[v] = {}
        val_ebd_dic[v] = {}
        val_tags_dic[v] = {}
        src_fold_root_path = root_path + r"\fold_set_" + v
        # test_folders = ['absent', 'present', 'reverse0.8', 'reverse0.9', 'reverse1.0', 'reverse1.1',
        #                 'reverse1.2', 'time_stretch0.8', 'time_stretch0.9', 'time_stretch1.1', 'time_stretch1.2']
        test_folders = ['absent', 'present']
        for folder in test_folders:
            val_tags_dic[v][folder] = np.load(npy_path_padded +
                                              f"\\{folder}_tags_norm01_fold{v}.npy", allow_pickle=True)
            val_labels_dic[v][folder] = np.load(npy_path_padded +
                                                f"\\{folder}_labels_norm01_fold{v}.npy", allow_pickle=True)
            val_index_dic[v][folder] = np.load(npy_path_padded +
                                               f"\\{folder}_index_norm01_fold{v}.npy", allow_pickle=True)
            val_mel_dic[v][folder] = np.load(npy_path_padded +
                                             f"\\{folder}_mel_norm01_fold{v}.npy", allow_pickle=True)

    return train_tags_dic, train_labels_dic, train_index_dic, train_mel_dic, val_tags_dic, val_labels_dic, val_index_dic, val_mel_dic


def fold5_dataloader(set_path, train_folder, test_folder, data_augmentation, set_name):
    """组合特征并且返回features，label，index"""
    # train_folders = ['absent', 'present', 'reverse0.8', 'reverse0.9', 'reverse1.0', 'reverse1.1', 'reverse1.2',
    #                  'time_stretch0.8', 'time_stretch0.9', 'time_stretch1.1', 'time_stretch1.2']
    train_folders = ['absent', 'present']
    train_tag_dic, train_labels_dic, train_index_dic, train_mel_dic, test_tag_dic, test_labels_dic, test_index_dic, test_mel_dic = get_features(
        set_path, train_folder, test_folder, set_name, train_folders)
    # train_wav_dic={}

    if data_augmentation:
        # train_wav = np.vstack(
        #     (
        #
        #         train_wav_dic[train_folder[0]][train_folders[0]],
        #         train_wav_dic[train_folder[0]][train_folders[1]],
        #         train_wav_dic[train_folder[0]][train_folders[2]],
        #         train_wav_dic[train_folder[0]][train_folders[3]],
        #         train_wav_dic[train_folder[0]][train_folders[4]],
        #         train_wav_dic[train_folder[0]][train_folders[5]],
        #         train_wav_dic[train_folder[0]][train_folders[6]],
        #         train_wav_dic[train_folder[0]][train_folders[7]],
        #         train_wav_dic[train_folder[0]][train_folders[8]],
        #         train_wav_dic[train_folder[0]][train_folders[9]],
        #         train_wav_dic[train_folder[0]][train_folders[10]],
        #         train_wav_dic[train_folder[1]][train_folders[0]],
        #         train_wav_dic[train_folder[1]][train_folders[1]],
        #         train_wav_dic[train_folder[1]][train_folders[2]],
        #         train_wav_dic[train_folder[1]][train_folders[3]],
        #         train_wav_dic[train_folder[1]][train_folders[4]],
        #         train_wav_dic[train_folder[1]][train_folders[5]],
        #         train_wav_dic[train_folder[1]][train_folders[6]],
        #         train_wav_dic[train_folder[1]][train_folders[7]],
        #         train_wav_dic[train_folder[1]][train_folders[8]],
        #         train_wav_dic[train_folder[1]][train_folders[9]],
        #         train_wav_dic[train_folder[1]][train_folders[10]],
        #         train_wav_dic[train_folder[2]][train_folders[0]],
        #         train_wav_dic[train_folder[2]][train_folders[1]],
        #         train_wav_dic[train_folder[2]][train_folders[2]],
        #         train_wav_dic[train_folder[2]][train_folders[3]],
        #         train_wav_dic[train_folder[2]][train_folders[4]],
        #         train_wav_dic[train_folder[2]][train_folders[5]],
        #         train_wav_dic[train_folder[2]][train_folders[6]],
        #         train_wav_dic[train_folder[2]][train_folders[7]],
        #         train_wav_dic[train_folder[2]][train_folders[8]],
        #         train_wav_dic[train_folder[2]][train_folders[9]],
        #         train_wav_dic[train_folder[2]][train_folders[10]],
        #         train_wav_dic[train_folder[3]][train_folders[0]],
        #         train_wav_dic[train_folder[3]][train_folders[1]],
        #         train_wav_dic[train_folder[3]][train_folders[2]],
        #         train_wav_dic[train_folder[3]][train_folders[3]],
        #         train_wav_dic[train_folder[3]][train_folders[4]],
        #         train_wav_dic[train_folder[3]][train_folders[5]],
        #         train_wav_dic[train_folder[3]][train_folders[6]],
        #         train_wav_dic[train_folder[3]][train_folders[7]],
        #         train_wav_dic[train_folder[3]][train_folders[8]],
        #         train_wav_dic[train_folder[3]][train_folders[9]],
        #         train_wav_dic[train_folder[3]][train_folders[10]]
        #     )
        # )
        train_label = np.hstack(
            (
                train_labels_dic[train_folder[0]][train_folders[0]],
                train_labels_dic[train_folder[0]][train_folders[1]],
                train_labels_dic[train_folder[0]][train_folders[2]],
                train_labels_dic[train_folder[0]][train_folders[3]],
                train_labels_dic[train_folder[0]][train_folders[4]],
                train_labels_dic[train_folder[0]][train_folders[5]],
                train_labels_dic[train_folder[0]][train_folders[6]],
                train_labels_dic[train_folder[0]][train_folders[7]],
                train_labels_dic[train_folder[0]][train_folders[8]],
                train_labels_dic[train_folder[0]][train_folders[9]],
                train_labels_dic[train_folder[0]][train_folders[10]],
                train_labels_dic[train_folder[1]][train_folders[0]],
                train_labels_dic[train_folder[1]][train_folders[1]],
                train_labels_dic[train_folder[1]][train_folders[2]],
                train_labels_dic[train_folder[1]][train_folders[3]],
                train_labels_dic[train_folder[1]][train_folders[4]],
                train_labels_dic[train_folder[1]][train_folders[5]],
                train_labels_dic[train_folder[1]][train_folders[6]],
                train_labels_dic[train_folder[1]][train_folders[7]],
                train_labels_dic[train_folder[1]][train_folders[8]],
                train_labels_dic[train_folder[1]][train_folders[9]],
                train_labels_dic[train_folder[1]][train_folders[10]],
                train_labels_dic[train_folder[2]][train_folders[0]],
                train_labels_dic[train_folder[2]][train_folders[1]],
                train_labels_dic[train_folder[2]][train_folders[2]],
                train_labels_dic[train_folder[2]][train_folders[3]],
                train_labels_dic[train_folder[2]][train_folders[4]],
                train_labels_dic[train_folder[2]][train_folders[5]],
                train_labels_dic[train_folder[2]][train_folders[6]],
                train_labels_dic[train_folder[2]][train_folders[7]],
                train_labels_dic[train_folder[2]][train_folders[8]],
                train_labels_dic[train_folder[2]][train_folders[9]],
                train_labels_dic[train_folder[2]][train_folders[10]],
                train_labels_dic[train_folder[3]][train_folders[0]],
                train_labels_dic[train_folder[3]][train_folders[1]],
                train_labels_dic[train_folder[3]][train_folders[2]],
                train_labels_dic[train_folder[3]][train_folders[3]],
                train_labels_dic[train_folder[3]][train_folders[4]],
                train_labels_dic[train_folder[3]][train_folders[5]],
                train_labels_dic[train_folder[3]][train_folders[6]],
                train_labels_dic[train_folder[3]][train_folders[7]],
                train_labels_dic[train_folder[3]][train_folders[8]],
                train_labels_dic[train_folder[3]][train_folders[9]],
                train_labels_dic[train_folder[3]][train_folders[10]]
            )
        )
        train_index = np.hstack(
            (
                train_index_dic[train_folder[0]][train_folders[0]],
                train_index_dic[train_folder[0]][train_folders[1]],
                train_index_dic[train_folder[0]][train_folders[2]],
                train_index_dic[train_folder[0]][train_folders[3]],
                train_index_dic[train_folder[0]][train_folders[4]],
                train_index_dic[train_folder[0]][train_folders[5]],
                train_index_dic[train_folder[0]][train_folders[6]],
                train_index_dic[train_folder[0]][train_folders[7]],
                train_index_dic[train_folder[0]][train_folders[8]],
                train_index_dic[train_folder[0]][train_folders[9]],
                train_index_dic[train_folder[0]][train_folders[10]],
                train_index_dic[train_folder[1]][train_folders[0]],
                train_index_dic[train_folder[1]][train_folders[1]],
                train_index_dic[train_folder[1]][train_folders[2]],
                train_index_dic[train_folder[1]][train_folders[3]],
                train_index_dic[train_folder[1]][train_folders[4]],
                train_index_dic[train_folder[1]][train_folders[5]],
                train_index_dic[train_folder[1]][train_folders[6]],
                train_index_dic[train_folder[1]][train_folders[7]],
                train_index_dic[train_folder[1]][train_folders[8]],
                train_index_dic[train_folder[1]][train_folders[9]],
                train_index_dic[train_folder[1]][train_folders[10]],
                train_index_dic[train_folder[2]][train_folders[0]],
                train_index_dic[train_folder[2]][train_folders[1]],
                train_index_dic[train_folder[2]][train_folders[2]],
                train_index_dic[train_folder[2]][train_folders[3]],
                train_index_dic[train_folder[2]][train_folders[4]],
                train_index_dic[train_folder[2]][train_folders[5]],
                train_index_dic[train_folder[2]][train_folders[6]],
                train_index_dic[train_folder[2]][train_folders[7]],
                train_index_dic[train_folder[2]][train_folders[8]],
                train_index_dic[train_folder[2]][train_folders[9]],
                train_index_dic[train_folder[2]][train_folders[10]],
                train_index_dic[train_folder[3]][train_folders[0]],
                train_index_dic[train_folder[3]][train_folders[1]],
                train_index_dic[train_folder[3]][train_folders[2]],
                train_index_dic[train_folder[3]][train_folders[3]],
                train_index_dic[train_folder[3]][train_folders[4]],
                train_index_dic[train_folder[3]][train_folders[5]],
                train_index_dic[train_folder[3]][train_folders[6]],
                train_index_dic[train_folder[3]][train_folders[7]],
                train_index_dic[train_folder[3]][train_folders[8]],
                train_index_dic[train_folder[3]][train_folders[9]],
                train_index_dic[train_folder[3]][train_folders[10]]
            )
        )
        train_mel = np.hstack(
            (
                train_mel_dic[train_folder[0]][train_folders[0]],
                train_mel_dic[train_folder[0]][train_folders[1]],
                train_mel_dic[train_folder[0]][train_folders[2]],
                train_mel_dic[train_folder[0]][train_folders[3]],
                train_mel_dic[train_folder[0]][train_folders[4]],
                train_mel_dic[train_folder[0]][train_folders[5]],
                train_mel_dic[train_folder[0]][train_folders[6]],
                train_mel_dic[train_folder[0]][train_folders[7]],
                train_mel_dic[train_folder[0]][train_folders[8]],
                train_mel_dic[train_folder[0]][train_folders[9]],
                train_mel_dic[train_folder[0]][train_folders[10]],
                train_mel_dic[train_folder[1]][train_folders[0]],
                train_mel_dic[train_folder[1]][train_folders[1]],
                train_mel_dic[train_folder[1]][train_folders[2]],
                train_mel_dic[train_folder[1]][train_folders[3]],
                train_mel_dic[train_folder[1]][train_folders[4]],
                train_mel_dic[train_folder[1]][train_folders[5]],
                train_mel_dic[train_folder[1]][train_folders[6]],
                train_mel_dic[train_folder[1]][train_folders[7]],
                train_mel_dic[train_folder[1]][train_folders[8]],
                train_mel_dic[train_folder[1]][train_folders[9]],
                train_mel_dic[train_folder[1]][train_folders[10]],
                train_mel_dic[train_folder[2]][train_folders[0]],
                train_mel_dic[train_folder[2]][train_folders[1]],
                train_mel_dic[train_folder[2]][train_folders[2]],
                train_mel_dic[train_folder[2]][train_folders[3]],
                train_mel_dic[train_folder[2]][train_folders[4]],
                train_mel_dic[train_folder[2]][train_folders[5]],
                train_mel_dic[train_folder[2]][train_folders[6]],
                train_mel_dic[train_folder[2]][train_folders[7]],
                train_mel_dic[train_folder[2]][train_folders[8]],
                train_mel_dic[train_folder[2]][train_folders[9]],
                train_mel_dic[train_folder[2]][train_folders[10]],
                train_mel_dic[train_folder[3]][train_folders[0]],
                train_mel_dic[train_folder[3]][train_folders[1]],
                train_mel_dic[train_folder[3]][train_folders[2]],
                train_mel_dic[train_folder[3]][train_folders[3]],
                train_mel_dic[train_folder[3]][train_folders[4]],
                train_mel_dic[train_folder[3]][train_folders[5]],
                train_mel_dic[train_folder[3]][train_folders[6]],
                train_mel_dic[train_folder[3]][train_folders[7]],
                train_mel_dic[train_folder[3]][train_folders[8]],
                train_mel_dic[train_folder[3]][train_folders[9]],
                train_mel_dic[train_folder[3]][train_folders[10]]
            )
        )
        train_tag = np.hstack(
            (
                train_tag_dic[train_folder[0]][train_folders[0]],
                train_tag_dic[train_folder[0]][train_folders[1]],
                train_tag_dic[train_folder[0]][train_folders[2]],
                train_tag_dic[train_folder[0]][train_folders[3]],
                train_tag_dic[train_folder[0]][train_folders[4]],
                train_tag_dic[train_folder[0]][train_folders[5]],
                train_tag_dic[train_folder[0]][train_folders[6]],

            )
        )

    else:
        # train_wav = np.vstack(
        #     (
        #         train_wav_dic[train_folder[0]][train_folders[0]],
        #         train_wav_dic[train_folder[0]][train_folders[1]],
        #         train_wav_dic[train_folder[1]][train_folders[0]],
        #         train_wav_dic[train_folder[1]][train_folders[1]],
        #         train_wav_dic[train_folder[2]][train_folders[0]],
        #         train_wav_dic[train_folder[2]][train_folders[1]],
        #         train_wav_dic[train_folder[3]][train_folders[0]],
        #         train_wav_dic[train_folder[3]][train_folders[1]]
        #     )
        # )
        train_label = np.hstack(
            (
                train_labels_dic[train_folder[0]][train_folders[0]],
                train_labels_dic[train_folder[0]][train_folders[1]],
                train_labels_dic[train_folder[1]][train_folders[0]],
                train_labels_dic[train_folder[1]][train_folders[1]],
                train_labels_dic[train_folder[2]][train_folders[0]],
                train_labels_dic[train_folder[2]][train_folders[1]],
                train_labels_dic[train_folder[3]][train_folders[0]],
                train_labels_dic[train_folder[3]][train_folders[1]]

            )
        )
        train_index = np.hstack(
            (
                train_index_dic[train_folder[0]][train_folders[0]],
                train_index_dic[train_folder[0]][train_folders[1]],
                train_index_dic[train_folder[1]][train_folders[0]],
                train_index_dic[train_folder[1]][train_folders[1]],
                train_index_dic[train_folder[2]][train_folders[0]],
                train_index_dic[train_folder[2]][train_folders[1]],
                train_index_dic[train_folder[3]][train_folders[0]],
                train_index_dic[train_folder[3]][train_folders[1]]
            )
        )
        train_mel = np.concatenate(
            (
                train_mel_dic[train_folder[0]][train_folders[0]],
                train_mel_dic[train_folder[0]][train_folders[1]],
                train_mel_dic[train_folder[1]][train_folders[0]],
                train_mel_dic[train_folder[1]][train_folders[1]],
                train_mel_dic[train_folder[2]][train_folders[0]],
                train_mel_dic[train_folder[2]][train_folders[1]],
                train_mel_dic[train_folder[3]][train_folders[0]],
                train_mel_dic[train_folder[3]][train_folders[1]]

            ), axis=0
        )
        train_tag = np.concatenate(
            (
                train_tag_dic[train_folder[0]][train_folders[0]],
                train_tag_dic[train_folder[0]][train_folders[1]],
                train_tag_dic[train_folder[1]][train_folders[0]],
                train_tag_dic[train_folder[1]][train_folders[1]],
                train_tag_dic[train_folder[2]][train_folders[0]],
                train_tag_dic[train_folder[2]][train_folders[1]],
                train_tag_dic[train_folder[3]][train_folders[0]],
                train_tag_dic[train_folder[3]][train_folders[1]]
            )
        )

    # test_wav = np.vstack(
    #     (
    #         test_wav_dic[test_folder[0]]['absent'],
    #         test_wav_dic[test_folder[0]]['present'],
    #     )
    # )
    val_label = np.hstack(
        (
            test_labels_dic[test_folder[0]]['absent'],
            test_labels_dic[test_folder[0]]['present'],
        )
    )
    val_index = np.hstack(
        (
            test_index_dic[test_folder[0]]['absent'],
            test_index_dic[test_folder[0]]['present'],
        )
    )
    val_mel = np.concatenate(
        (
            test_mel_dic[test_folder[0]]['absent'],
            test_mel_dic[test_folder[0]]['present']
        ), axis=0
    )
    val_tag = np.concatenate(
        (
            test_tag_dic[test_folder[0]]['absent'],
            test_tag_dic[test_folder[0]]['present'],
        )
    )

    return train_mel, train_label, train_index, train_tag, val_mel, val_label, val_index, val_tag  # , test_ebd
