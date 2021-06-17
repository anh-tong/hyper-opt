
from typing import Callable
from model import BaseHyperOptModel
from torch import autograd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, RMSprop
from torch.nn.utils.convert_parameters import parameters_to_vector
from utils import fixed_point, gvp, jvp, neumann, conjugate_gradient



class BaseHyperOptimizer():
    
    def __init__(
        self,
        parameters,
        hyper_parameters,
        base_optimizer='SGD',
        default=dict(lr=0.01),
        use_gauss_newton=True,
        ) -> None:
        
        self.parameters = parameters
        self.hyper_parameters = hyper_parameters
        
        if base_optimizer == 'SGD':
            self.hyper_optim = SGD(self.hyper_parameters, **default)
        elif base_optimizer == 'RMSprop':
            self.hyper_optim = RMSprop(self.hyper_parameters, **default)
        else:
            ValueError("Only consider base_optimizer with SGD and RMSprop optimizer")
        
        self.use_gauss_newton = use_gauss_newton
        
        self.build_inverse_hvp()
        
    def build_inverse_hvp(self, **kwargs):
        """ Subclass will customize the way how inverse hessian vector product is computed
        
            the function should provide
                self.inverse_hvp as a callable function
                self.inverse_hvp_kwargs as additional arguments for self.inverse_hvp
        """
        raise NotImplementedError
    
    def step(self, train_loss, val_loss, train_logit=None, verbose=False):
        
        if self.use_gauss_newton:
            assert train_logit is not None
        
        ## v1: in the algorithm
        dval_dparam = autograd.grad(
            outputs=val_loss,
            inputs=self.parameters,
            retain_graph=True)
        
        dtrain_dparam = autograd.grad(
            train_loss,
            self.parameters,
            create_graph=True, # take higher-order derivative
            retain_graph=True  # keep the computation graph of train_loss
            )
                
        def hessian_vector_product(v):
            return jvp(dtrain_dparam, self.parameters, vector=v)
        
        def gauss_newton_vector_product(v):
            return gvp(train_loss, train_logit, self.parameters, vector=v)
        
        ## v2: inverse hessian product
        if self.use_gauss_newton:
            mvp = gauss_newton_vector_product
        else:
            mvp = hessian_vector_product
        inv_hvp = self.inverse_hvp(mvp, vector=parameters_to_vector(dval_dparam), **self.inverse_hvp_kwargs)
        
        ## v3: indirect gradient
        indirect_gradient = autograd.grad(
            parameters_to_vector(dtrain_dparam),
            self.hyper_parameters,
            grad_outputs=inv_hvp
        )
        
        ## direct gradient. this is for the general case. 
        direct_gradient = autograd.grad(
            val_loss,
            self.hyper_parameters, 
            allow_unused=True # allow that hyperparameters no need to appear in the validationn loss
            )
        direct_gradient = [torch.zeros_like(p) if d is None else d 
                            for d, p in zip(direct_gradient, self.hyper_parameters)]
        
        total_gradient = [direct - indirect for direct, indirect in zip(direct_gradient, indirect_gradient)]
        
        # clipping gradient to prevent gradient explosion (maybe due to computation instability)
        clip_grad_norm_(total_gradient, max_norm=1.)  # TODO: make it configurable
        
        # update grad and perform gradient descent step
        self.hyper_optim.zero_grad()
        for p, direct, indirect in zip(self.hyper_parameters, direct_gradient, indirect_gradient):
            p.grad = direct - indirect
            
        if verbose:
            # print out some quanities
            direct_gradient_ = parameters_to_vector(direct_gradient)
            indirect_gradient_ = parameters_to_vector(indirect_gradient)
            direct_norm = torch.norm(direct_gradient_)
            indirect_norm = torch.norm(indirect_gradient_)
            print(f"Direct norm: {direct_norm.item():.3f} \t Indirect norm: {indirect_norm.item():.3f} ")
            
        self.hyper_optim.step()
        
class NeumannHyperOptimizer(BaseHyperOptimizer):
    
    def __init__(self,
                 parameters, 
                 hyper_parameters, 
                 base_optimizer='SGD', 
                 default=dict(lr=0.01),
                 use_gauss_newton=True) -> None:
        super().__init__(parameters,
                         hyper_parameters,
                         base_optimizer=base_optimizer,
                         default=default,
                         use_gauss_newton=use_gauss_newton)
    
    def build_inverse_hvp(self, lr=0.01, truncate_iter=5):
        kwargs = dict(lr=lr, truncate_iter=truncate_iter)
        self.inverse_hvp = neumann
        self.inverse_hvp_kwargs = kwargs
    
class ConjugateHyperOptimizer(BaseHyperOptimizer):
    
    def __init__(self, 
                 parameters,
                 hyper_parameters,
                 base_optimizer='SGD',
                 default=dict(lr=0.01)) -> None:
        # Always use Gauss-Newton Hessian because it's possitive definite
        super().__init__(parameters, hyper_parameters, base_optimizer=base_optimizer, default=default, use_gauss_newton=True)
        
    def build_inverse_hvp(self, num_iter=20):
        self.inverse_hvp_kwargs = dict(num_iter=num_iter)
        self.inverse_hvp = conjugate_gradient  

class FixedPointHyperOptimizer(BaseHyperOptimizer):
    
    def __init__(self,
                 parameters, 
                 hyperparmeters,
                 base_optimizer='SGD', 
                 default=dict(lr=0.01),
                 use_gauss_newton=True) -> None:
        super().__init__(parameters,
                         hyperparmeters, 
                         base_optimizer=base_optimizer, 
                         default=default,
                         use_gauss_newton=use_gauss_newton)
        
    def build_inverse_hvp(self, num_iter=20, lr=0.1):
        self.inverse_hvp_kwargs = dict(num_iter=num_iter, lr=lr)
        self.inverse_hvp = fixed_point


class BaseHyperOptimizer_v2():
    
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
                return gvp(train_loss, train_logit, self.parameters, vector=v, retain_graph=False)
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
                    grad_outputs=v
                )
                
            def gauss_newton_vector_product(v):
                return gvp(train_loss, train_logit, self.parameters, vector=v, retain_graph=True)  # this case must retain graph
        
        if self.use_gauss_newton:
            mvp = gauss_newton_vector_product
        else:
            mvp = hessian_vector_product
        
        p = self.solve(A=mvp, x=dval_dparam)   
        p = v = dval_dparam
        for _ in range(self.K):
            output = mvp(v)
            v = [v_ + self.inner_lr * o_ for v_, o_ in zip(v, output)]
            p = [v_ + p_ for v_, p_ in zip(v, p)]
        
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

    def solve(self, A: Callable, x):
        
        p = v = x
        for _ in range(self.K):
            output = A(v)
            v = [v_ + self.inner_lr * o_ for v_, o_ in zip(v, output)]
            p = [v_ + p_ for v_, p_ in zip(v, p)]
            # early stopping?
            
        return p
    
class NeumannHyperOptimizer_v2(BaseHyperOptimizer_v2):
    
    def __init__(
        self, 
        parameters, hyper_parameters,
        base_optimizer='SGD', 
        default=dict(lr=0.1),
        use_gauss_newton=True, 
        stochastic=True) -> None:
        super().__init__(parameters, hyper_parameters, base_optimizer=base_optimizer, default=default,
                         use_gauss_newton=use_gauss_newton, stochastic=stochastic)
    
    def solve(self, A: Callable, x):
        
        p = v = x
        for _ in range(self.K):
            output = A(v)
            v = [v_ + self.inner_lr * o_ for v_, o_ in zip(v, output)]
            p = [v_ + p_ for v_, p_ in zip(v, p)]
            # early stopping?
            
        return p
    
class FixedPointHyperOptimizer_v2(BaseHyperOptimizer_v2):
    
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

class HyperOptOptimizer_v2(BaseHyperOptimizer):
    
    """This implememetation is roughly the same with the implementation of https://github.com/prolearner/hypertorch/blob/master/hypergrad/hypergradients.py
    
        This is used to compared with our implemetation
    """
    
    def __init__(self,
                 parameters, 
                 hyper_parameters, 
                 base_optimizer='SGD',
                 default=dict(lr=0.1),
                 use_gauss_newton=False) -> None:
        super().__init__(parameters,
                         hyper_parameters,
                         base_optimizer=base_optimizer,
                         default=default, 
                         use_gauss_newton=use_gauss_newton)
        
    def step(self, train_loss, val_loss, train_logit, verbose):
        K = 20
        inner_lr=0.1
        
        dval_dparam = autograd.grad(val_loss, self.parameters, retain_graph=True)
        
        dtrain_dparam = autograd.grad(train_loss, self.parameters, create_graph=True)
        w_mapped = [p -inner_lr * g for p, g in zip(self.parameters, dtrain_dparam)]
        
        
        
        # start fixed point iteration
        vs = [torch.zeros_like(w) for w in self.parameters]
        vs_vec = parameters_to_vector(vs)
        for k in range(K):
            vs_prev_vec = vs_vec
            vs = autograd.grad(w_mapped,
                               self.parameters,
                               grad_outputs=vs,
                               retain_graph=True)
            vs = [v + g for v, g in zip(vs, dval_dparam)]
            
            vs_vec = parameters_to_vector(vs)
            print("norm", (vs_vec - vs_prev_vec).norm().item())
        
        indirect = autograd.grad(w_mapped, 
                                 self.hyper_parameters, 
                                 grad_outputs=vs,
                                 allow_unused=True)
        direct = autograd.grad(val_loss,
                               self.hyper_parameters,
                               allow_unused=True)
        
        total_grad = [d + i if d is not None else i for d, i in zip(direct, indirect)]
        
        self.hyper_optim.zero_grad()
        for p, g in zip(self.hyper_parameters, total_grad):
            p.grad = g
        self.hyper_optim.step()
        
    def build_inverse_hvp(self, **kwargs):
        pass

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
    
    
    

        
    
    