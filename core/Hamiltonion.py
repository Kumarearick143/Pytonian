import torch
import torch.nn as nn

class LearnableHamiltonian(nn.Module):
    def __init__(self, size):
        super().__init__()
        # Hermitian parameterized matrix as Hamiltonian
        self.H_real = nn.Parameter(torch.randn(size, size))
        self.H_imag = nn.Parameter(torch.randn(size, size))
        
    def forward(self, psi):
        # Construct Hermitian matrix H = A + iB, H = H^dagger
        H = self.H_real + 1j * self.H_imag
        H = (H + H.conj().t()) / 2
        return H @ psi
