# pytron_qft/utils/qmath.py
import torch

def commutator(A, B):
    """Compute commutator [A, B] = AB - BA"""
    return torch.matmul(A, B) - torch.matmul(B, A)

def anticommutator(A, B):
    """Compute anticommutator {A, B} = AB + BA"""
    return torch.matmul(A, B) + torch.matmul(B, A)

def braket(psi, operator, phi):
    """Compute <ψ|O|φ>"""
    return torch.vdot(psi, torch.matmul(operator, phi))

def partial_trace(rho, dims, keep):
    """Partial trace over specified subsystems"""
    keep = sorted(keep)
    dims_keep = [dims[i] for i in keep]
    dims_trace = [d for i, d in enumerate(dims) if i not in keep]
    
    rho_reshaped = rho.view(*dims, *dims)
    for i in sorted([i for i in range(len(dims)) if i not in keep], reverse=True):
        rho_reshaped = rho_reshaped.trace(i, i + len(dims))
    
    return rho_reshaped.view(torch.prod(torch.tensor(dims_keep)), 
                            torch.prod(torch.tensor(dims_keep)))