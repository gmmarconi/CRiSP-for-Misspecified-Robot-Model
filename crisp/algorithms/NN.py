import torch as th
import torch.nn as nn
from torch.nn.modules import Module
import numpy as np
import torch.optim as optim
import time
import copy
from pathlib import Path

class fc_nn(Module):
    """
    Class for a simple 5 layer neural network in PyTorch
    """

    def __init__(self, input_dim, output_dim, device):
        super(fc_nn, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = th.tanh(self.fc2(x))
        x = th.tanh(self.fc3(x))
        x = th.tanh(self.fc4(x))
        x = th.tanh(self.fc5(x))
        x = th.tanh(self.fc6(x))
        x = self.fc7(x)
        return x

    def predict(self, X, y0=None, yte=None, is_sequence=False):
        with th.no_grad():
            if X.ndim > 1 :
                preds = np.full((X.shape[0], self.output_dim), np.inf)
                for idx, point in enumerate(X):
                    preds[idx] = (self.forward(th.tensor(point, dtype=th.float).to(self.device)).cpu().numpy())
                return preds, {}
            else:
                return self.forward(th.tensor(X, dtype=th.float).to(self.device)).cpu().numpy(), {}

    def evaluate(self, xtest, ytest):
        with th.no_grad():
            preds = self.predict(xtest)[0]
            return np.mean((ytest - preds)**2)

def trainer(xtrain, ytrain, model, device, lr=0.0001, max_epochs=10, output_folder=Path(','), save_name=None, xval=None, yval=None):
    """
    Trainer function for simple_nn
    :param xtrain: x of training set
    :param ytrain: y of training se
    :param model: any pre-existing model to use as a starting training point
    :param device: device where to store the model, CPU or GPU
    :param max_epochs: maximum number of training peochs (1 epoch = 1 pass over the whole training set)
    :param output_folder: wheere to save the models during training
    :param save_name: name for saving the model
    :param xval:
    :param yval:
    :return: the model with minimum loss on the validation set
    """
    if save_name is None:
        save_name = 'NN'
    ntrain = xtrain.shape[0]
    best_val_loss = np.inf

    xtrain_th = th.from_numpy(xtrain).float().to(device)
    ytrain_th = th.from_numpy(ytrain).float().to(device)
    model.to(device)
    criterion = nn.MSELoss()

    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    print(f"Training NN model with {ntrain} training points")
    t0 = time.time()
    epoch = 0
    while epoch < max_epochs:
        epoch_loss, net, optimizer = train_epoch(xtrain_th, ytrain_th, model, optimizer,
                                                criterion, b_size=200, device=device,)
        if (epoch+1) % 50 == 0 or epoch == 0:
            if xval is not None and yval is not None:
                val_loss = model.evaluate(xval, yval)
                if val_loss < best_val_loss:
                    th.save(model.state_dict(), output_folder / (save_name+'_best.pth'))
                    best_val_loss = val_loss
                print(f'[Epoch: {epoch + 1}/{max_epochs}] Tr loss: {epoch_loss / xtrain.shape[0]:7.5f}\t Val loss: {val_loss:7.5f}')
            else:
                if epoch == 0:
                    th.save(model.state_dict(), output_folder /  (save_name+'_best.pth'))
                print(f'[Epoch: {epoch + 1}/{max_epochs}, Points:{xtrain.shape[0]}] loss: {epoch_loss / xtrain.shape[0]:7.5f}')
        epoch += 1

    best_on_val = copy.deepcopy(model)
    best_on_val.load_state_dict(th.load(output_folder / (save_name+'_best.pth')))
    t1 = time.time()
    print(f"Training completed in {(t1 - t0) / 60} mins")
    return model, best_on_val

def train_epoch(xtrain, ytrain, net, optimizer, criterion, b_size, device):
    """
    Trains a single epoch of the supplied model
    :return: the loss at the end of the epoch on the training set, the model and the optimizer used for training
    """
    epoch_loss = 0.0
    indexes = th.randperm(xtrain.shape[0])

    for idx in range(0, xtrain.shape[0], b_size):
        inputs = xtrain[indexes[idx:min(idx + b_size, xtrain.shape[0])]].to(device)
        labels = ytrain[indexes[idx:min(idx + b_size, xtrain.shape[0])]]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss, net, optimizer