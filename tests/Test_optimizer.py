# tests/test_optimizer.py
import torch
import pytest
from pytron_qft.optim.path_integral import PathIntegralOptimizer

def test_path_integral_optimizer():
    model = torch.nn.Linear(2, 1)
    optimizer = PathIntegralOptimizer(model.parameters(), lr=0.01, n_paths=4, temp=0.1)
    
    # Dummy loss
    def loss_fn(model):
        x = torch.randn(2)
        return model(x).sum()
    
    # Original parameters
    original_params = [p.clone() for p in model.parameters()]
    
    # Optimization step
    optimizer.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    optimizer.step(loss_fn)
    
    # Check parameters changed
    for p_orig, p_new in zip(original_params, model.parameters()):
        assert not torch.allclose(p_orig, p_new)