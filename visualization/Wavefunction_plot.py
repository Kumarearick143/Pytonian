# pytron_qft/visualization/wavefunction_plot.py
import matplotlib.pyplot as plt
import numpy as np

def plot_wavefunction(psi, title="Wavefunction Visualization", ax=None):
    """Visualize wavefunction magnitude and phase"""
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    magnitude = np.abs(psi)
    phase = np.angle(psi)
    
    # Magnitude plot
    ax[0].plot(magnitude)
    ax[0].set_title(f"{title} - Magnitude")
    ax[0].set_xlabel("State Index")
    ax[0].set_ylabel("|ψ|")
    
    # Phase plot
    ax[1].plot(phase)
    ax[1].set_title(f"{title} - Phase")
    ax[1].set_xlabel("State Index")
    ax[1].set_ylabel("Phase (radians)")
    ax[1].set_ylim(-np.pi, np.pi)
    
    return ax