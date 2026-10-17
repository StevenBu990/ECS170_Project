'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import pandas as pd


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data from', self.dataset_source_file_name)
        path = self.dataset_source_folder_path + self.dataset_source_file_name

        df = pd.read_csv(path, header=None)
        print(df.shape)

        y = df.iloc[:, 0]
        y = y.to_numpy(dtype=np.int64)

        X = df.iloc[:, 1:]
        X = X.to_numpy(dtype=np.float32)

        X /= 255.0

        return {'X': X, 'y': y}