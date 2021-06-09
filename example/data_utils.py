from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

def load_mnist(batch_size, num_train=50000):
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_dataset = datasets.MNIST(root=".",
                           train=True,
                           download=True,
                           transform=transform)
    train_dataset.data = train_dataset.data[:num_train, ...]
    train_dataset.targets = train_dataset.targets[:num_train, ...]
    
    val_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    val_dataset.data = val_dataset.data[num_train:, ...]
    val_dataset.targets = val_dataset.targets[num_train:,...]
    
    test_dataset = datasets.MNIST(root=".",
                                  train=False,
                                  download=False,
                                  transform=transform)
        
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=False)
    
    return train_loader, val_loader, test_loader

class InfiniteDataLoader():
    
    def __init__(self, data_loader, device) -> None:
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.epoch_elapsed = 0
        self.device = device
    
    def next_batch(self):
        
        try:
            x, y = self.data_iter.next()
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            x, y = self.data_iter.next()
            self.epoch_elapsed += 1

        x, y = x.to(self.device), y.to(self.device)
        return x, y
            


if __name__ == "__main__":
    
    train, val, test = load_mnist(batch_size=1)
    print(len(train))
    print(len(val))
    
    