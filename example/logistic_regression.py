import sys
sys.path.append("/home/anhth/projects/hyper-opt")
from model import BaseHyperOptModel
from hyper_opt import ConjugateHyperOptimizer, NeumannHyperOptimizer
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn((input_dim, 1)))
    
    def forward(self, x):
        return x @ self.w
    

class LRHyperOptModel(BaseHyperOptModel):
    
    def __init__(self, input_dim) -> None:
        network = LogisticRegression(input_dim)
        criterion = nn.BCEWithLogitsLoss()
        super().__init__(network, criterion)
        
        self.hparams = nn.Parameter(torch.ones(input_dim,1))
        
    @property
    def hyper_parameters(self):
        return [self.hparams]
    
    def regularizer(self):
        return 0.5 * (self.network.w.t() @ torch.diag(self.hparams.squeeze())) @ self.network.w
    


n, d = 1000, 20
outer_steps = 1000
T, K = 100, 10

w_oracle = torch.randn(d, 1)
x = torch.randn(n, d)
y = x @ w_oracle + 0.1*torch.randn(n, 1)
y = (y >0.).float()

x_train, y_train = x[:n//2], y[:n//2]
x_val, y_val = x[n//2:], y[n//2:]

model = LRHyperOptModel(input_dim=d)
train_loss,_ = model.train_loss(x_train, y_train)
val_loss = model.validation_loss(x_val, y_val)

w_optimizer = torch.optim.SGD(model.parameters, lr=0.1)
hyper_optimizer = ConjugateHyperOptimizer(model.parameters, model.hyper_parameters, default=dict(lr=1., momentum=.9))
hyper_optimizer.build_inverse_hvp(num_iter=K)
# hyper_optimizer = NeumannHyperOptimizer(model.parameters, model.hyper_parameters, default=dict(lr=1., momentum=.9), use_gauss_newton=False)
# hyper_optimizer.build_inverse_hvp(lr=1., truncate_iter=K)


for o_step in range(outer_steps):
    
    # inner optimizer
    for t in range(T):
        w_optimizer.zero_grad()
        loss, _ = model.train_loss(x_train, y_train)
        loss.backward()
        w_optimizer.step()
    
    # outer optimizer
    train_loss, train_logit = model.train_loss(x_train, y_train)
    val_loss = model.validation_loss(x_val, y_val)
    hyper_optimizer.step(train_loss, val_loss, train_logit)
    model.hyper_parameters[0].data.clamp_(min=1e-8)
    
    if o_step % 10 == 0:
        print(f"Outer step {o_step:3d} \t Val Loss: {val_loss.item():.3f}")