
from typing import Callable
from model import BaseHyperOptModel
from torch import autograd
import torch
from torch.optim import SGD, RMSprop
from torch.nn.utils.convert_parameters import parameters_to_vector
from utils import gvp


class BaseHyperOptimizer():
    
    def __init__(
        self, 
        parameters,
        hyper_parameters,
        base_optimizer='SGD',
        default=dict(lr=0.1), 
        use_gauss_newton=True,
        stochastic=True) -> None:
        
        self.parameters = parameters
        self.hyper_parameters = hyper_parameters
        
        if base_optimizer == 'SGD':
            self.hyper_optim = SGD(self.hyper_parameters, **default)
        elif base_optimizer == 'RMSprop':
            self.hyper_optim = RMSprop(self.hyper_parameters, **default)
        else:
            ValueError("Only consider base_optimizer with SGD and RMSprop optimizer")
        
        self.use_gauss_newton = use_gauss_newton
        self.stochastic = stochastic
        
        self.set_kwargs()
        
    def set_kwargs(self, inner_lr=0.1, K=20, **kwargs):
        self.inner_lr = inner_lr
        self.K = K
        
    
    def step(self, train_loss_fun, val_loss, verbose=False):
        
        dval_dparam = autograd.grad(
            val_loss,
            self.parameters,
            retain_graph=True
        )
        
        if self.stochastic:
            def hessian_vector_product(v):
                # evaluate train_loss on minibatch data every time this function call
                train_loss, _ = train_loss_fun()
                dtrain_dparam = autograd.grad(
                    train_loss,
                    self.parameters,
                    create_graph=True
                )
                
                return autograd.grad(
                    dtrain_dparam,
                    self.parameters,
                    grad_outputs=v
                )
            
            def gauss_newton_vector_product(v):
                train_loss, train_logit = train_loss_fun()
                return gvp(train_loss, train_logit, self.parameters, vector=v)
        else:
            # train_loss is just evaluated once
            train_loss, train_logit = train_loss_fun()
            def hessian_vector_product(v):
                dtrain_dparam = autograd.grad(
                    train_loss,
                    self.parameters,
                    create_graph=True,
                    retain_graph=True # this case must retain graph
                )
                
                return autograd.grad(
                    dtrain_dparam,
                    self.parameters,
                    grad_outputs=v,
                    retain_graph=True
                )
                
            def gauss_newton_vector_product(v):
                return gvp(train_loss, train_logit, self.parameters, vector=v, retain_graph=True)  # this case must retain graph
        
        if self.use_gauss_newton:
            mvp = gauss_newton_vector_product
        else:
            mvp = hessian_vector_product
        
        p = self.solve(A=mvp, x=dval_dparam)   
        
        if self.stochastic:
            train_loss = - self.inner_lr * train_loss_fun()[0]
        else:
            train_loss = - self.inner_lr * train_loss
            
        minus_dtrain_dparam = autograd.grad(
            train_loss,
            self.parameters,
            create_graph=True
        )
        
        indirect = autograd.grad(
            minus_dtrain_dparam,
            self.hyper_parameters,
            grad_outputs=p
        )
        
        direct = autograd.grad(
            val_loss,
            self.hyper_parameters,
            allow_unused=True
        )
        
        total_grad = [d + i if d is not None else i for d, i in zip(direct, indirect)]
        
        self.hyper_optim.zero_grad()
        for p, g in zip(self.hyper_parameters, total_grad):
            p.grad = g
        self.hyper_optim.step()

    def solve(self, A: Callable, b):
        """
        Solving a linear system Ax = b
        
        Subclass will implement this method.
        Args:
            A (Callable): A vector-matrix product 
            b ([type]): vector
        """
        
        raise NotImplementedError
    
class NeumannHyperOptimizer(BaseHyperOptimizer):
    
    def __init__(
        self, 
        parameters, hyper_parameters,
        base_optimizer='SGD', 
        default=dict(lr=0.1),
        use_gauss_newton=True, 
        stochastic=True) -> None:
        super().__init__(parameters, hyper_parameters, base_optimizer=base_optimizer, default=default,
                         use_gauss_newton=use_gauss_newton, stochastic=stochastic)
    
    def solve(self, A: Callable, b):
        """
        
        
        Returns:
            [type]: [description]
        """
        
        p = v = b
        for _ in range(self.K):
            output = A(v)
            v = [v_ + self.inner_lr * o_ for v_, o_ in zip(v, output)]
            p = [v_ + p_ for v_, p_ in zip(v, p)]
            # early stopping?
            
        return p
    
class FixedPointHyperOptimizer(BaseHyperOptimizer):
    
    def __init__(
        self, 
        parameters, hyper_parameters,
        base_optimizer='SGD', 
        default=dict(lr=0.1),
        use_gauss_newton=True, 
        stochastic=True) -> None:
        super().__init__(parameters, hyper_parameters, base_optimizer=base_optimizer, default=default,
                         use_gauss_newton=use_gauss_newton, stochastic=stochastic)
        
    def set_kwargs(self, inner_lr=0.1, K=20, eta=0.9):
        super().set_kwargs(inner_lr=inner_lr, K=K)
        self.eta = eta
    
    def solve(self, A: Callable, x):
        
        v = x
        for _ in range(self.K):
            output = A(v)
            hat_phi = [v_ - self.inner_lr * o_ + x_ for v_, o_, x_ in zip(v, output, x)]
            v = [(1.-self.eta)* v_ + self.eta * phi for v_, phi in zip(v, hat_phi)]

            # TODO: early stopping
        return v
    
class ConjugateHyperOptimizer(BaseHyperOptimizer):
    
    def __init__(
        self, 
        parameters, 
        hyper_parameters, 
        base_optimizer='SGD', 
        default=dict(lr=0.1), 
        use_gauss_newton=True,
        stochastic=True) -> None:
        super().__init__(parameters, hyper_parameters, base_optimizer=base_optimizer, default=default,
                         use_gauss_newton=use_gauss_newton, stochastic=stochastic)
        
    def solve(self, A: Callable, b):
        # TODO: implement this
        raise NotImplementedError


if __name__ == "__main__":
    
    """Simple unit test"""
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleModel(nn.Module):

        def __init__(self):
            super().__init__()
            
            self.net = nn.Sequential(*[
                nn.Linear(1,2),
                nn.Sigmoid(),
                nn.Linear(2,1)]
            )
            
            self.hyper_param = nn.Parameter(torch.tensor([0.1]))
            
        def forward(self, x):
            return self.net(x)
        
        def loss(self, x, y):
            return F.mse_loss(self(x), y)
        
        def weight_penalty(self):
            ret = 0.
            for param in self.net.parameters():
                ret += torch.norm(param)
            ret *= self.hyper_param.squeeze()
            return ret
    
    
    model = SimpleModel()
    
    x_train = torch.randn(2, 1)
    y_train = torch.randn(2, 1)
    
    x_val, y_val = torch.randn(2,1), torch.randn(2,1)
    
    train_logit = model(x_train)
    train_loss = F.mse_loss(train_logit, y_train) + model.weight_penalty()
    val_loss = model.loss(x_val, y_val)
    
    # neumann case
    optimizer = NeumannHyperOptimizer(
        parameters=list(model.net.parameters()),
        hyper_parameters=[model.hyper_param], 
        use_gauss_newton=False)
    optimizer.step(train_loss, val_loss, train_logit, verbose=True)
    
    # conjugate case
    # optimizer = ConjugateHyperOptimizer(
    #     parameters=list(model.net.parameters()),
    #     hyper_parameters=[model.hyper_param])
    # optimizer.step(train_loss, val_loss, train_logit, verbose=True)
    
    
    

        
    
    