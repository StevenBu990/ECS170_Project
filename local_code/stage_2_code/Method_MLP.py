'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    optimizer_name = 'adam'
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 128

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        
        ## WE WANT TO EXPERIMENT WITH 256 and 128 in output of Linear
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(256, 128)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))

        # output layer: 10 raw scores, one per digit (no softmax)
        y_pred = self.fc_layer_3(h2)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


        loss_function = nn.CrossEntropyLoss()

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
                optimizer.zero_grad()   # erasse correctiosn form last batch
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