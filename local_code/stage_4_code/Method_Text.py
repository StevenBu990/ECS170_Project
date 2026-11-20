
'''
Shared base class for the Stage 4 text models: training (DataLoader + gradient clipping + GPU),
validation, testing, and run(). Each model inherits from this and only defines its layers and forward().
'''

from local_code.base_class.method import method
from local_code.stage_4_code.Text_Dataset import Text_Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader

# optional, for GPU training
def pick_device():
    if torch.backends.mps.is_available():   # Apple GPU
        return torch.device('mps')
    if torch.cuda.is_available():           # NVIDIA GPU
        return torch.device('cuda')
    return torch.device('cpu')

class Method_Text(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64
    optimizer_name = 'adam'

    def __init__(self, mName, mDescription, device='auto'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = pick_device() if device == 'auto' else torch.device(device)

    def train_model(self, X, y, X_val=None, y_val=None):
        self.to(self.device)                # move every layer's weights to the GPU
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        loader = DataLoader(Text_Dataset(X, y), batch_size=self.batch_size, shuffle=True)
        n=len(X)
        self.history = {'epoch': [], 'loss': [], 'acc': [], 'val_acc': []}

        for epoch in range(self.max_epoch):
            self.train()
            total_loss, correct = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                logits = self.forward(xb)
                loss = loss_function(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()

            epoch_loss = total_loss / n
            epoch_acc = correct / n
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss)
            self.history['acc'].append(epoch_acc)
            print('Epoch', epoch+1, 'loss:', round(epoch_loss, 4), 'acc:', round(epoch_acc, 4))

            if X_val is not None:
                val_acc = (self.test(X_val) == y_val).mean()   # fraction of val images reviews correct
                self.history['val_acc'].append(val_acc)
                print('     val acc:', round(val_acc, 4))
                
    @torch.no_grad()
    def test(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.long)
        preds = []
        for i in range(0, len(X), 1000):
            logits = self.forward(X[i:i + 1000].to(self.device))
            preds.append(logits.argmax(1).cpu())
        return torch.cat(preds).numpy()

    def run(self):
        print('method running on', self.device, '...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'], self.data['val']['X'], self.data['val']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

    @torch.no_grad()
    def generate(self, three_word_ids, sos_id, eos_id, max_new_ids=300):
        self.eval()
        ids = [sos_id] + list(three_word_ids)
        x = torch.tensor([ids], device=self.device)

        logits, hidden = self(x)
        next_logits = logits[0, -1]

        for _ in range(max_new_ids):
            probs = torch.softmax(next_logits, dim=-1)
            nxt_id = torch.multinomial(probs, 1).item()

            if nxt_id == eos_id:
                break

            ids.append(nxt_id)
            logits, hidden = self(torch.tensor([[nxt_id]], device=self.device), hidden)
            next_logits = logits[0, -1]

        return ids[1:]