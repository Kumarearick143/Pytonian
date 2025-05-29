```markdown
<!-- docs/usage_guide.md -->
# Pytron-QFT User Guide

## Installation
```bash
pip install pytron_qft
```

## Basic Usage

### Define a Quantum Field Model
```python
from pytron_qft import QuantumField, OperatorProduct, GeometricMeasurement

class FieldNet(nn.Module):
    def __init__(self):
        self.field = QuantumField([2, 3])
        self.evolve = OperatorProduct(256)
        self.measure = GeometricMeasurement(10)
    
    def forward(self, x):
        psi = self.field(x)
        psi = self.evolve(psi)
        return self.measure(psi)
```

### Train with Path Integral Optimization
```python
from pytron_qft.optim import PathIntegralOptimizer

model = FieldNet()
optimizer = PathIntegralOptimizer(model.parameters(), n_paths=8)

for batch in loader:
    loss = model(batch)
    optimizer.step(loss)
```

### Visualize Wavefunction
```python
from pytron_qft.visualization import plot_wavefunction

plot_wavefunction(model.measure.psi)
```

## Examples
See `examples/` directory for:
- `train_fieldnet.py`: MNIST classification
- `train_topogan.py`: Material generation
- `train_quantumgpt.py`: Text generation
```
