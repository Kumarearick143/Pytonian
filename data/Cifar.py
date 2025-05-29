import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .field_ops import spatial_to_spectral, renormalize_field

class QuantumCIFAR(Dataset):
    def __init__(self, root='./data', train=True, spectral=True, 
                 renormalize=True, cutoff=1e-3, download=True):
        """
        Quantum Field Representation of CIFAR-10
        
        Args:
            root (str): Root directory of dataset
            train (bool): Load training set if True, else test set
            spectral (bool): Convert to spectral representation
            renormalize (bool): Apply renormalization cutoff
            cutoff (float): Renormalization cutoff value
            download (bool): Download dataset if not available
        """
        self.spectral = spectral
        self.renormalize = renormalize
        self.cutoff = cutoff
        
        # Base transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                (0.2470, 0.2435, 0.2616))
        ])
        
        # Load CIFAR-10 dataset
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=download, 
            transform=transform
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert to complex representation
        complex_img = torch.view_as_complex(
            torch.stack((image, torch.zeros_like(image)), dim=-1)
        )
        
        # Apply spectral transformation
        if self.spectral:
            complex_img = spatial_to_spectral(complex_img)
            
        # Renormalize small amplitudes
        if self.renormalize:
            complex_img = renormalize_field(complex_img, self.cutoff)
            
        return complex_img, label
    
    def get_class_names(self):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

def create_cifar_loaders(batch_size=64, spectral=True, renormalize=True):
    """
    Create quantum CIFAR-10 data loaders
    
    Returns:
        train_loader, test_loader, class_names
    """
    train_set = QuantumCIFAR(train=True, spectral=spectral, renormalize=renormalize)
    test_set = QuantumCIFAR(train=False, spectral=spectral, renormalize=renormalize)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader, train_set.get_class_names()

# Example usage:
# train_loader, test_loader, class_names = create_cifar_loaders()
# for images, labels in train_loader:
#     print(images.shape)  # [batch, 3, 32, 32] complex
#     break