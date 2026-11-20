'''
RNN text classifier: embedding -> recurrent layer (RNN, LSTM or GRU) -> pooled hidden states -> linear (2 classes).
Options: bidirectional reading, 'last' or 'max' pooling over time, and dropout.
Inherits training, validation and testing from Method_Text.
'''

from local_code.stage_4_code.Method_Text import Method_Text
import torch
from torch import nn

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class Method_RNN_Classifier(Method_Text):

    def __init__(self, mName, mDescription, vocab_size, device='auto',
                 cell='rnn', embed_dim=100, hidden_dim=128, num_layers=1,
                 bidirectional=False, pooling='last', dropout=0.0):
        Method_Text.__init__(self, mName, mDescription, device)
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)    # word ID -> vector
        self.rnn = CELLS[cell](input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)    # forward and backward states are concatenated
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 2)                       # pooled representation -> 2 scores (neg, pos)

    def forward(self, x):                         # x: (batch, seq_len) word IDs
        emb = self.drop(self.embedding(x))        # -> (batch, seq_len, embed_dim)
        out, _ = self.rnn(emb)                    # -> (batch, seq_len, out_dim), hidden state at every word
        if self.pooling == 'max':
            rep = out.max(dim=1).values           # strongest value of each feature over all words
        else:
            rep = out[:, -1, :]                   # hidden state at the last word (reviews are left-padded)
        return self.fc(self.drop(rep))
