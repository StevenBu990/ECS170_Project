'''
RNN language model for joke generation: embedding -> recurrent layer (RNN, LSTM or GRU) -> linear over the vocabulary.
Trained to predict the next token at every position (padding ignored in the loss).
generate() starts from <sos> + three words and keeps predicting until <eos>.
'''

from local_code.stage_4_code.Method_Text import Method_Text
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class Method_RNN_Generator(Method_Text):
    max_epoch = 100
    learning_rate = 2e-3
    batch_size = 32

    def __init__(self, mName, mDescription, vocab_size, device='auto',
                 cell='lstm', embed_dim=128, hidden_dim=256, num_layers=1):
        Method_Text.__init__(self, mName, mDescription, device)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = CELLS[cell](input_size=embed_dim, hidden_size=hidden_dim,
                               num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)          # hidden state -> a score for every word

    def forward(self, x, hidden=None):                       # x: (batch, seq_len) token IDs
        out, hidden = self.rnn(self.embedding(x), hidden)    # out: (batch, seq_len, hidden_dim)
        return self.fc(out), hidden                          # logits: (batch, seq_len, vocab_size)

    def train_model(self, X, y):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)  # don't learn to predict padding
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                            batch_size=self.batch_size, shuffle=True)
        self.history = {'epoch': [], 'loss': [], 'acc': []}

        for epoch in range(self.max_epoch):
            self.train()
            total_loss, total_tokens, correct = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, _ = self.forward(xb)
                loss = loss_function(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                mask = yb != 0                                   # real tokens only
                n_tokens = mask.sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                correct += ((logits.argmax(-1) == yb) & mask).sum().item()

            epoch_loss, epoch_acc = total_loss / total_tokens, correct / total_tokens
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss)
            self.history['acc'].append(epoch_acc)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print('Epoch', epoch + 1, 'loss:', round(epoch_loss, 4), 'next-word acc:', round(epoch_acc, 4))

    @torch.no_grad()
    def generate(self, start_ids, sos_id, eos_id, max_new_ids=40, greedy=True):
        '''Start from <sos> + the given word IDs and keep predicting the next word until <eos>.
        greedy=True always takes the most likely word; False samples from the predicted probabilities.'''
        self.eval()
        ids = [sos_id] + list(start_ids)
        logits, hidden = self(torch.tensor([ids], device=self.device))
        next_logits = logits[0, -1]
        for _ in range(max_new_ids):
            if greedy:
                nxt_id = int(next_logits.argmax())
            else:
                nxt_id = int(torch.multinomial(torch.softmax(next_logits, dim=-1), 1))
            if nxt_id == eos_id:
                break
            ids.append(nxt_id)
            logits, hidden = self(torch.tensor([[nxt_id]], device=self.device), hidden)
            next_logits = logits[0, -1]
        return ids[1:]                                       # drop <sos>

    def run(self):
        print('method running on', self.device, '...')
        print('--start training...')
        self.train_model(self.data['X'], self.data['y'])
        return self.history
