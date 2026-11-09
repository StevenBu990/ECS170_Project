'''
Loads the Stage 3 pickle datasets (MNIST, ORL, CIFAR) into
PyTorch-ready arrays of shape (N, channels, height, width).
'''

from local_code.base_class.dataset import dataset
import numpy as np
import pickle

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None    # 'MNIST', 'ORL', or 'CIFAR'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data from', self.dataset_source_file_name, '...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        return {'train': self.convert(raw['train']), 'test': self.convert(raw['test'])}

    def convert(self, instances):
        # list of {'image', 'label'} dicts -> two numpy arrays
        X = np.array([inst['image'] for inst in instances], dtype=np.float32)
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)
        name = self.dataset_source_file_name

        # scale every pixel to the range 0-1
        X /= 255.0

        # reshape X to (N, C, H, W), depending on name:
        #   'MNIST' -> add a channel axis
        #   'ORL'   -> keep only channel 0, then add a channel axis
        #   'CIFAR' -> move the channel axis to position 1
        if name == 'MNIST':
            X = X[:, np.newaxis, :, :]
        elif name == 'ORL':
            X = X[:, :, :, 0]
            X = X[:, np.newaxis, :, :]
        elif name == 'CIFAR':
            X = X.transpose(0, 3, 1, 2)

        # if name is 'ORL', shift labels from 1-40 to 0-39
        if name == 'ORL':
            y -= 1

        return {'X': X, 'y': y}