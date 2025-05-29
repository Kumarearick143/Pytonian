# tests/test_quantumfield.py
import torch
import pytest
from pytron_qft.core.quantum_field import QuantumField

def test_quantum_field_forward():
    field = QuantumField(field_dims=[2, 3])
    input_tensor = torch.randn(5, 5, dtype=torch.cfloat)
    output = field(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.is_complex(output)
    
def test_renormalization():
    field = QuantumField(field_dims=[2], cutoff=0.1)
    # Set a mode to be below cutoff
    field.modes['k_0'].data = torch.tensor([0.05+0j, 0.15+0j], dtype=torch.cfloat)
    field.renormalize()
    assert torch.allclose(field.modes['k_0'].data[0], torch.tensor(0j))
    assert not torch.allclose(field.modes['k_0'].data[1], torch.tensor(0j))