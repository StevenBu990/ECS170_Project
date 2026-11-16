'''
RNN text classifier: embedding -> recurrent layer (RNN, LSTM or GRU) -> last hidden state -> linear (2 classes).
Inherits training, validation and testing from Method_Text.
'''

from local_code.stage_4_code.Method_Text import Method_Text
import torch
from torch import nn

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class Method_RNN_Classifier(Method_Text):

    def __init__(self, mName, mDescription, vocab_size, device='auto',
                 cell='rnn', embed_dim=100, hidden_dim=128, num_layers=1):
        Method_Text.__init__(self, mName, mDescription, device)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)   # word ID -> vector
        self.rnn = CELLS[cell](input_size=embed_dim, hidden_size=hidden_dim,
                               num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):                 # x: (batch, seq_len) word IDs
        emb = self.embedding(x)           # -> (batch, seq_len, embed_dim)
        out, _ = self.rnn(emb)            # out: (batch, seq_len, hidden_dim), the hidden state at every step
        last = out[:, -1, :]
        return self.fc(last)