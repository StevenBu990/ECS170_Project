
'''
PyTorch Dataset wrapping encoded text (word-ID sequences) and labels, so a DataLoader can batch and shuffle them.
'''

import torch
from torch.utils.data import Dataset

class Text_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])