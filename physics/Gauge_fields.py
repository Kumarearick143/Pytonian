import torch
import torch.nn as nn

class GaugeFieldEncoder(nn.Module):
    def __init__(self, input_dim, gauge_group_dim=3):
        """
        Encode classical data into gauge field representations.
        
        Args:
            input_dim: number of input features
            gauge_group_dim: dimension of the gauge group representation (e.g. SU(2) -> 3)
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, gauge_group_dim)
    
    def forward(self, x):
        """
        Transforms input data to gauge field representation space.
        
        Args:
            x: input tensor shape (batch_size, input_dim)
        
        Returns:
            gauge field tensor (batch_size, gauge_group_dim)
        """
        return self.encoder(x)
