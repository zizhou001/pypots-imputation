import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict['X'])

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx], dtype=torch.float32) for key, value in self.data_dict.items()}


def create_dict(features, labels=None, masks=None, batch_size=32):
    # 创建数据字典
    data_dict = {'X': features}
    if labels is not None:
        data_dict['y'] = labels
    if masks is not None:
        data_dict['mask'] = masks

    return data_dict
