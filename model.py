from network import UNet
import torch
import torch.nn as nn
from torch.nn.modules.container import ParameterList

class BaseHyperOptModel(nn.Module):
    
    # TODO: how to implement this like decorator pattern. If we have multiple approaches 
    def __init__(self, network, criterion) -> None:
        super().__init__()
        
        self.network = network
        self.criterion = criterion
    
    @property
    def parameters(self):
        return list(self.network.parameters())
    
    @property
    def hyper_parameters(self):
        raise NotImplementedError
    
    def train_loss(self, x, y):
        
        x = self.data_augment(x, y)
        
        logit = self.network(x)
        loss = self.criterion(logit, y)
        regularizer = self.regularizer()
        
        return loss + regularizer, logit
    
    def validation_loss(self, x, y):
        
        logit = self.network(x)
        loss = self.criterion(logit, y)
        return loss
    
    def data_augment(self, x, y):
        """
        Overwrite this to perform data_augmentation task
        """
        return x
    
    def regularizer(self):
        """
        Overwrite this for customizing regularizer
        """
        return 0.
        
class L2HyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)
        self.l2 = nn.Parameter(torch.tensor([-10.]))
        
    @property
    def hyper_parameters(self):
        return [self.l2]
    
    def regularizer(self):
        
        ret = 0.
        for param in self.parameters:
            ret += torch.sum(torch.mul(param, param))
        return ret * (10** self.l2)
    

class AllL2HyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)
        
        weight_decay = []
        for param in self.network.parameters():
            l2 = nn.Parameter(torch.ones_like(param)*(-10.))
            weight_decay.append(l2)
        
        self.weight_decay = ParameterList(weight_decay)
    
    @property
    def hyper_parameters(self):
        return list(self.weight_decay.parameters())
    
    def regularizer(self):
        ret = 0.
        for weight_decay, param in zip(self.weight_decay, self.parameters):
            l2 = weight_decay.exp()
            ret += torch.sum(l2 * param **2)
        return ret
    

class UNetAugmentHyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)
        
        self.augment_net = UNet()
        
    
    @property
    def hyper_parameters(self):
        return list(self.augment_net.parameters())
    
    def data_augment(self, x, y):
        return self.augment_net(x, y)
            
    
class ReweightHyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)