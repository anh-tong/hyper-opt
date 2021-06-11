import argparse
import sys
import torch
sys.path.append("/home/anhth/projects/hyper-opt")

import torch.nn as nn
from torch.optim import Adam

from network import MLP, SimpleModel
from model import AllL2HyperOptModel, L2HyperOptModel
from hyper_opt import ConjugateHyperOptimizer, NeumannHyperOptimizer


from example.data_utils import InfiniteDataLoader, load_mnist

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--hyper_lr", type=float, default=0.1)
parser.add_argument("--clip", type=float, default=0.25)
parser.add_argument("--device", type=str, default="cuda:5")

args = parser.parse_args()
device = torch.device(args.device)

# get data
train_loader, val_loader, test_loader = load_mnist(batch_size=args.batch_size)
train_iter = InfiniteDataLoader(train_loader, device)
val_iter = InfiniteDataLoader(val_loader, device)

# neural network model
model = MLP(num_layers=5, input_shape=(28, 28))
# model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# wrap the neural net with hyperparameter model
h_model = AllL2HyperOptModel(model, criterion)
h_model.to(device=device)

# init two optimizers: one for neural network weights, the other for hypeparmeter
weight_optimizer = Adam(h_model.parameters, lr=args.lr)
hyper_optimizer = NeumannHyperOptimizer(h_model.parameters,
                                        h_model.hyper_parameters,
                                        base_optimizer='SGD',
                                        default=dict(lr=args.hyper_lr),
                                        use_gauss_newton=True)
hyper_optimizer.build_inverse_hvp(lr=1.)

# hyper_optimizer = ConjugateHyperOptimizer(h_model.parameters,h_model.hyper_parameters, default=dict(lr=args.hyper_lr))


def evaluate():
    model.eval()
    with torch.no_grad():
        
        total_loss, correct = 0., 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y)
            total_loss += loss.item()
            pred = torch.argmax(logit, dim=1).float()
            correct += (y == pred).float().sum().item()
    model.train()
    acc = float(correct) / len(test_loader.dataset) 
    
    return acc, total_loss
    

def train():
    
    counter = 0
    while train_iter.epoch_elapsed <= 10:
        
        model.train()
        train_x, train_y = train_iter.next_batch()
        train_loss, train_logit = h_model.train_loss(train_x, train_y)
        weight_optimizer.zero_grad()
        train_loss.backward() # retain graph so we can reuse this train loss
        weight_optimizer.step()
        
        val_x, val_y = val_iter.next_batch()
        val_loss = h_model.validation_loss(val_x, val_y)
        if train_iter.epoch_elapsed >=1:
            train_loss, train_logit = h_model.train_loss(train_x, train_y)
            hyper_optimizer.step(train_loss, val_loss, train_logit, verbose=True)
        
        
        if counter % 10 == 0 and counter > 0:
            eval_acc, eval_loss = evaluate()

            train_loss = train_loss.item()
            val_loss = val_loss.item() if val_loss is not None else 0.
            print(f"Iter {counter:5d} | train loss {train_loss:5.2f} | \
                val loss {val_loss:5.2f} | Test loss {eval_loss:5.2f}| Test acc {eval_acc:2.4f}")
            # print(h_model.hyper_parameters[0].cpu().data.numpy())
        
        
        counter += 1
train()