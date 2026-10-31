
'''
Shared base class for the Stage 3 CNNs: training (DataLoader + GPU), testing, and run().
Each dataset-specific CNN inherits from this and only defines its layers and forward().
'''

from local_code.base_class.method import method
from local_code.stage_3_code.Image_Dataset import Image_Dataset
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

class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64
    optimizer_name = 'adam'

    def __init__(self, mName, mDescription, device='auto'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = pick_device() if device == 'auto' else torch.device(device)

    def train_model(self, X, y):
        self.to(self.device)                # move every layer's weights to the GPU
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        loader = DataLoader(Image_Dataset(X, y), batch_size=self.batch_size, shuffle=True)
        n=len(X)
        self.history = {'epoch': [], 'loss': [], 'acc': []}

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
                optimizer.step()

                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()

            epoch_loss = total_loss / n
            epoch_acc = correct / n
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss)
            self.history['acc'].append(epoch_acc)
            print('Epoch', epoch+1, 'loss:', round(epoch_loss, 4), 'acc:', round(epoch_acc, 4))
    
    @torch.no_grad()
    def test(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        preds = []
        for i in range(0, len(X), 1000):
            logits = self.forward(X[i:i + 1000].to(self.device))
            preds.append(logits.argmax(1).cpu())
        return torch.cat(preds).numpy()
    
    def run(self):
        print('method running on', self.device, '...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}