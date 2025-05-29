# pytron_qft/core/measurement.py
import torch

class GeometricMeasurement(torch.nn.Module):
    def __init__(self, output_dim, curvature=0.1):
        super().__init__()
        self.R = nn.Parameter(torch.tensor(curvature))
        self.projection = nn.Linear(output_dim, output_dim, dtype=torch.cfloat)
        
    def forward(self, ψ):
        """Measurement with manifold curvature"""
        # Project onto measurement basis
        ψ_proj = self.projection(ψ)
        # Compute probabilities with curvature
        probs = ψ_proj.abs().square()
        curved_probs = probs ** (self.R + 1)
        return curved_probs / curved_probs.sum(dim=-1, keepdim=True)