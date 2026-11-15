'''
Loads the IMDB review dataset (train/test x pos/neg folders of .txt files),
cleans each review, builds the vocabulary from the training set only,
and encodes every review as a fixed-length sequence of word IDs.
'''

import os
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Cleaner import clean_review
from local_code.stage_4_code.Vocabulary import Vocabulary


class Dataset_Loader_Review(dataset):
    data = None
    dataset_source_folder_path = None    # path to the text_classification folder
    max_len = 200                        # words kept per review (after cleaning)
    min_freq = 5                         # a word must appear 5+ times in training to get its own ID
    vocab = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def read_split(self, split):
        '''Read one split ('train' or 'test') -> (list of cleaned word lists, list of labels).'''
        texts, labels = [], []
        for label_name, label in [('neg', 0), ('pos', 1)]:
            folder = os.path.join(self.dataset_source_folder_path, split, label_name)
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith('.txt'):          # skip .DS_Store and other non-review files
                    continue
                with open(os.path.join(folder, fname), encoding='utf-8') as f:
                    raw = f.read()
                    texts.append(clean_review(raw))
                    labels.append(label)

        return texts, labels

    def load(self):
        print('loading reviews (50,000 files, this takes a little while)...')
        train_texts, train_y = self.read_split('train')
        test_texts, test_y = self.read_split('test')

        self.vocab = Vocabulary(self.min_freq)
        self.vocab.build(train_texts)

        print('vocabulary size:', len(self.vocab))

        X_train = np.array([self.vocab.encode(t, self.max_len) for t in train_texts], dtype=np.int64)
        # TODO 3: X_test, the same way, from test_texts
        X_test = np.array([self.vocab.encode(t, self.max_len) for t in test_texts], dtype=np.int64)

        return {'train': {'X': X_train, 'y': np.array(train_y, dtype=np.int64)},
                'test':  {'X': X_test,  'y': np.array(test_y,  dtype=np.int64)}}