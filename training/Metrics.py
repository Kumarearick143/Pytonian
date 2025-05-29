# pytron_qft/training/metrics.py
import torch
import numpy as np
from gudhi import RipsComplex

def entanglement_entropy(wavefunction):
    """Calculate entanglement entropy of wavefunction"""
    density_matrix = torch.outer(wavefunction, wavefunction.conj())
    eigenvalues = torch.linalg.eigvalsh(density_matrix)
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-12))
    return entropy

def betti_numbers(wavefunction):
    """Compute topological features using persistent homology"""
    prob_dist = wavefunction.abs().square().cpu().numpy()
    # Create point cloud from probability distribution
    rips = RipsComplex(points=prob_dist)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    betti = st.betti_numbers()
    return betti[:3]  # Return first three Betti numbers