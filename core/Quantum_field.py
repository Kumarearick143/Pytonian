# pytron_qft/core/quantum_field.py
import torch
import torch.nn as nn

class QuantumField(nn.Module):
    def __init__(self, field_dims, cutoff=1e-3):
        super().__init__()
        self.field_dims = field_dims
        self.cutoff = nn.Parameter(torch.tensor(cutoff))
        self.modes = nn.ParameterDict({
            f'k_{i}': nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
            for i, dim in enumerate(field_dims)
        })
    
    def forward(self, x):
        # Apply spectral transformation
        for dim, param in zip(self.field_dims, self.modes.values()):
            # Fourier transform along specified dimension
            x = torch.fft.fft(x, dim=dim)
            # Apply mode filter
            x = self._apply_mode_filter(x, param, dim)
            # Inverse transform
            x = torch.fft.ifft(x, dim=dim)
        return x
    
    def _apply_mode_filter(self, x, modes, dim):
        shape = x.shape
        n = shape[dim]
        # Create filter from learned modes
        mode_filter = torch.zeros(n, dtype=torch.cfloat, device=x.device)
        mode_filter[:len(modes)] = modes
        # Apply along dimension
        return x * mode_filter.view(*[1]*dim, n, *[1]*(len(shape)-dim-1))

    def renormalize(self):
        """Apply renormalization to field modes"""
        for name, param in self.modes.items():
            magnitude = param.abs()
            phase = param.angle()
            # Apply cutoff
            magnitude = torch.where(magnitude < self.cutoff, 
                                  torch.zeros_like(magnitude), 
                                  magnitude)
            param.data = magnitude * torch.exp(1j * phase)