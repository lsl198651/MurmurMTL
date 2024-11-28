import torch
from torch.utils.data import Dataset


class DatasetClass(Dataset):
    """继承Dataset类，重写__getitem__和__len__方法
    添加get_idx方法，返回id
    input: wavlabel, wavdata, wavidx
    """

    # Initialize your data, download, etc.
    def __init__(self, features, wav_label, wav_index):
        self.data = torch.from_numpy(features)
        self.label = torch.from_numpy(wav_label)
        self.idx = torch.from_numpy(wav_index)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        label_item = self.label[index]
        idx_item = self.idx[index]
        return data_item.float(), label_item, idx_item

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)


class DatasetMTL(Dataset):
    """带有嵌入的数据集类"""

    def __init__(self, features, wav_label, wav_index, embedded):
        self.data = torch.from_numpy(features)
        self.label = torch.from_numpy(wav_label)
        self.idx = torch.from_numpy(wav_index)
        self.emb = torch.from_numpy(embedded)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        label_item = self.label[index]
        idx_item = self.idx[index]
        embedded = self.emb[index]
        # embeding = 1  # fake
        # wide_feat = hand_fea((data_item, 4000))
        return data_item.float(), label_item, idx_item, embedded

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)

class DatasetTest(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, wavlabel, wavdata, wavidx):
        # 直接传递data和label
        # self.len = wavlen
        # embeds = []
        # for embed in wavebd:
        #     embed = int(embed.split('.')[0])
        #     embeds.append(embed)
        # self.wavebd = embeds
        self.data = torch.from_numpy(wavdata)
        self.label = torch.from_numpy(wavlabel)
        self.id = torch.from_numpy(wavidx)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        dataitem = self.data[index]
        labelitem = self.label[index]
        iditem = self.id[index]
        # embeding = self.wavebd[index]
        embeding = 1  # fake
        # wide_feat = hand_fea((dataitem, 4000))
        return dataitem.float(), labelitem, iditem

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)
