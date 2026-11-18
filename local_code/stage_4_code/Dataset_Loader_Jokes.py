import csv
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Cleaner import clean_joke
from local_code.stage_4_code.Vocabulary import Vocabulary


class Dataset_Loader_Jokes(dataset):
    data = None
    dataset_source_folder_path = None
    max_len = 60
    min_freq = 5
    vocab = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        if self.data is not None:
            return self.data

        print('loading in jokes')
        jokes = []

        with open(self.dataset_source_folder_path,encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                jokes.append(clean_joke(row[1]))

        self.vocab = Vocabulary(self.min_freq)
        self.vocab.build(jokes)

        print('vocabulary size:', len(self.vocab))
        self.data = np.array([self.vocab.encode(j,self.max_len) for j in jokes], dtype=np.int64)

        return self.data