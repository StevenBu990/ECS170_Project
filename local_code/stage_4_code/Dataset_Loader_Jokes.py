'''
Loads the jokes CSV for text generation. Each joke is cleaned with clean_joke() (keeps stop words
and ? . ! , as tokens) and wrapped in <sos> ... <eos> markers so the model learns where jokes start and end.
Returns next-word training pairs: X = tokens[:-1], y = tokens[1:] (right-padded with <pad> = 0).
'''

import csv
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Cleaner import clean_joke
from local_code.stage_4_code.Vocabulary import Vocabulary


class Dataset_Loader_Jokes(dataset):
    data = None
    dataset_source_folder_path = None    # path to the jokes data file
    min_freq = 1                         # keep every word: the model must be able to reproduce whole jokes
    vocab = None
    jokes = None                         # cleaned jokes (lists of tokens), used for the comparison step
    max_len = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        if self.data is not None:
            return self.data

        print('loading jokes...')
        jokes = []
        with open(self.dataset_source_folder_path, encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)                       # skip the "ID","Joke" header
            for row in reader:
                if len(row) < 2:
                    continue
                tokens = clean_joke(row[1])
                if tokens:
                    jokes.append(tokens)
        self.jokes = jokes

        seqs = [['<sos>'] + j + ['<eos>'] for j in jokes]
        self.vocab = Vocabulary(self.min_freq)
        self.vocab.build(seqs)
        self.max_len = max(len(s) for s in seqs)
        print(f'{len(jokes)} jokes | vocabulary size: {len(self.vocab)} | longest joke: {self.max_len} tokens')

        # right-pad, so every joke starts at <sos> with a fresh hidden state (same as during generation)
        ids = []
        for s in seqs:
            seq_ids = [self.vocab.word2idx.get(w, 1) for w in s]
            ids.append(seq_ids + [0] * (self.max_len - len(seq_ids)))
        ids = np.array(ids, dtype=np.int64)

        self.data = {'X': ids[:, :-1], 'y': ids[:, 1:]}     # predict each next token
        return self.data
