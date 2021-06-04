
from typing import Callable, Iterable
import torch
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector

def jvp(outputs, inputs, vector, create_graph=False, retrain_graph=False, flatten=True):
    """ Jacobian vector product

    Args:
        outputs (Tensor or list of Tensor): ouputs of function
        inputs (list of Tensor): inputs of function
        vector (Tensor or list of Tensor): 
        create_graph (bool, optional): True if required to compute further higher deriveation. Defaults to False.
        retrain_graph (bool, optional): True if keep the graph for further autograd computation. Defaults to False.
        flatten (bool, optional): True if return a flat tensor else return a list of tensor. Defaults to True.
    """
    
    if isinstance(outputs, tuple):
        dummy = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    else:
        dummy = torch.zeros_like(outputs, requires_grad=True)
    
    jacobian = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=dummy,
        create_graph=True
    )
    
    if isinstance(vector, Iterable):
        vector = parameters_to_vector(vector)
    jacobian = parameters_to_vector(jacobian)
    Jv = grad(
        outputs=jacobian,
        inputs=dummy,
        grad_outputs=vector,
        create_graph=create_graph,
        retain_graph=retrain_graph
    )
    if flatten:
        return parameters_to_vector(Jv)
    else:
        return Jv

def hvp(loss,parameters,vector,retain_graph=True,flatten=True):
    """Hessian-vector product

    Args:
        loss (Tensor): loss value 
        parameters (list or tuple of Tensor): inputs of the loss
        vector (list or tuple of Tensor): vector in the product
        retain_graph (bool, optional): True if want to compute . Defaults to True.
        flatten (bool, optional): whether or not return a flat tensor. Defaults to True.
    """
    gradient = grad(
        loss,
        parameters,
        create_graph=True,
        retain_graph=retain_graph
    )
    
    return jvp(gradient, parameters, vector, retain_graph, flatten=flatten)

def gvp(loss, logit, parameters, vector, retain_graph=True, flatten=True):
    
    """Gauss-Newton Hessian-vector product
    """
    
    Jv = jvp(logit, parameters, vector, retrain_graph=retain_graph, flatten=True)
    
    HJv = hvp(loss, logit, vector=Jv, retain_graph=retain_graph, flatten=True)
    
    JHJv = grad(
        logit,
        parameters,
        grad_outputs=HJv.reshape_as(logit),
        retain_graph=retain_graph
    )
    
    if flatten:
        return parameters_to_vector(JHJv)
    else:
        return JHJv

def neumann(mvp: Callable, vector, lr=0.01, truncate_iter=5):
    """
    Inverse matrix product using neumann series expansion
    
    Args:
        mvp (a callable function): either Hessian vector product function or Gauss-Newton Hessian vector product
        vector (Tensor): the vector to multiply with inverse matrix
        lr (float, optional): learning rate . Defaults to 0.01.
        truncate_iter (int, optional): number of iterations to strop. Defaults to 5.
    """
    
    p = v = vector
    for i in range(truncate_iter):
        # TODO: consider damping
        output_grad = mvp(v)
        v = v - lr * output_grad
        p = p + v
    
    return p

def conjugate_gradient(mvp: Callable, vector, num_iter=20, eps=0.):
    """Conjugate gradient to find argmin of x^\top A x + 2b^\top x
    """
    
    x = vector.clone().detach()
    r = x - mvp(x)
    p = r.clone().detach()
    for _ in range(num_iter):
        Ap = mvp(p)
        alpha = (r @ r) / (p @ Ap + eps)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r + eps)
        p = r_new + beta * p
        r = r_new.clone().detach()
    
    return x    


if __name__ == "__main__":
    
    """Some unit test
    """
    
    def test_jvp():
        from torch.autograd import Variable
        
        x = Variable(torch.Tensor([1.]), requires_grad=True)
        y = Variable(torch.Tensor([0.]), requires_grad=True)
        
        f1 = x**2 + y**2
        f2 = x**2 - y**2
        
        outputs = (f1, f2)
        inputs = (x, y)
        vector = torch.ones((2,))
        
        result = jvp(outputs, inputs, vector, flatten=True, retrain_graph=True)
        print(result)
        result = jvp(outputs, inputs, vector, flatten=False, retrain_graph=True)
        print(result)
        
        # quickly 
        print(hvp(f2, inputs, vector, retain_graph=True))
        
    test_jvp()
        
    
    
    