# pytron_qft/data/field_ops.py
import torch
import torch.fft

def spatial_to_spectral(spatial_tensor):
    """Convert spatial data to spectral representation"""
    # FFT with shift for centered low frequencies
    spectral = torch.fft.fftshift(torch.fft.fft2(spatial_tensor), dim=(-2, -1))
    return spectral

def spectral_to_spatial(spectral_tensor):
    """Convert spectral data back to spatial representation"""
    spatial = torch.fft.ifft2(torch.fft.ifftshift(
        spectral_tensor, dim=(-2, -1)))
    return spatial.abs()

def renormalize_field(field, cutoff=1e-3):
    """Apply renormalization group flow to quantum field"""
    magnitude = field.abs()
    phase = field.angle()
    # Apply cutoff to small magnitudes
    magnitude = torch.where(magnitude < cutoff, torch.zeros_like(magnitude), magnitude)
    return magnitude * torch.exp(1j * phase)