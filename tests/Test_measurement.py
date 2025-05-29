# tests/test_measurement.py
import torch
from pytron_qft.core.measurement import GeometricMeasurement

def test_geometric_measurement():
    meas = GeometricMeasurement(output_dim=3, curvature=0.5)
    psi = torch.tensor([0.5+0j, 0.5j, 0.5], dtype=torch.cfloat)
    probs = meas(psi)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    assert probs.min() >= 0
    assert probs.max() <= 1