# pytron_qft/data/mnist.py
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from .field_ops import spatial_to_spectral

class QuantumMNIST:
    def __init__(self, batch_size=64, spectral=True):
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
        
        self.train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        self.test = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
        self.batch_size = batch_size
        self.spectral = spectral

    def get_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def quantum_encode(self, batch):
        images, labels = batch
        if self.spectral:
            # Convert to spectral representation
            return spatial_to_spectral(images), labels
        # Return as wavefunction (add imaginary component)
        return torch.view_as_complex(images.unsqueeze(-1)), labels