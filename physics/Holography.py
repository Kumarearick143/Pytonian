import torch
import torch.nn as nn

class AdSCFTInterface(nn.Module):
    def __init__(self, bulk_dim=16, boundary_dim=8):
        """
        Simple holographic mapping between bulk (AdS spacetime) and boundary (CFT) latent spaces.
        
        Args:
            bulk_dim: dimension of bulk spacetime representation
            boundary_dim: dimension of boundary conformal field theory representation
        """
        super().__init__()
        self.bulk_to_boundary = nn.Linear(bulk_dim, boundary_dim)
        self.boundary_to_bulk = nn.Linear(boundary_dim, bulk_dim)
    
    def encode(self, bulk_field):
        """
        Maps from bulk AdS space to boundary CFT representation.
        
        Args:
            bulk_field: tensor (batch_size, bulk_dim)
            
        Returns:
            boundary_field: tensor (batch_size, boundary_dim)
        """
        return self.bulk_to_boundary(bulk_field)
    
    def decode(self, boundary_field):
        """
        Maps from boundary CFT to bulk AdS representation.
        
        Args:
            boundary_field: tensor (batch_size, boundary_dim)
            
        Returns:
            bulk_field: tensor (batch_size, bulk_dim)
        """
        return self.boundary_to_bulk(boundary_field)
