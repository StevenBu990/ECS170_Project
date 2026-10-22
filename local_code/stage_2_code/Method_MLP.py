'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
from torch import nn

from local_code.base_class.method import method


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    optimizer_name = 'adam'
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, mName, mDescription, hidden_dims=(256, 128), loss_fn = 'cross-entropy'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.hidden_dims = hidden_dims
        self.loss_fn = loss_fn

        layers = []
        in_size = 784                       # input: one value per pixel
        for out_size in hidden_dims:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            in_size = out_size

        layers.append(nn.Linear(in_size, 10))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)                  # runs every layer in order

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def choose_loss(self):
        nll = nn.NLLLoss()
        lookup = {'cross-entropy': nn.CrossEntropyLoss(),
                  # NLLLoss expects log-probabilities, so turn the raw scores into those first
                  'negative-log': lambda logits, y: nll(torch.log_softmax(logits, dim=1), y),
                  'multi-margin': nn.MultiMarginLoss()}
        return lookup[self.loss_fn]

    def train_model(self, X, y):
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        loss_function = self.choose_loss()

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        n = len(X)
        self.history = {'epoch': [], 'loss': [], 'acc': []}

        for epoch in range(self.max_epoch):
            self.train()
            perm = torch.randperm(n)    # shuffle indices of 0-59999
            total_loss, correct = 0.0, 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]   # obtains next 128 shuffled indices
                xb, yb = X[idx], y[idx]             # batch's images and labels

                logits = self.forward(xb)
                loss = loss_function(logits, yb)
                optimizer.zero_grad()   # remove gradients from last batch
                loss.backward()         # work out correction for every weight
                optimizer.step()        # apply corrections
                

                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(1) == yb).sum().item()

            epoch_loss = total_loss / n
            epoch_acc = correct / n
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(epoch_loss)
            self.history['acc'].append(epoch_acc)
            print('Epoch', epoch + 1, 'loss:', round(epoch_loss, 4), 'acc:', round(epoch_acc, 4))

    
    @torch.no_grad()    # no learning during testing, so skip gradeint bookkeeping
    def test(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        return self.forward(X).argmax(1).numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}