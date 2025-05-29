# pytron_qft/visualization/manifold_plot.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_measurement_manifold(probs, curvature, title="Measurement Manifold"):
    """Visualize measurement manifold with curvature"""
    # Create parametric surface
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    
    # Apply curvature to sphere
    r = 1 + curvature * np.sin(3*theta) * np.cos(2*phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # Project probabilities onto manifold
    proj_x = probs[0] * x.mean()
    proj_y = probs[1] * y.mean()
    proj_z = probs[2] * z.mean()
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.5, rstride=1, cstride=1, 
                   color='b', edgecolor='k')
    
    # Plot measurement points
    ax.scatter(proj_x, proj_y, proj_z, s=100, c='r', marker='o')
    
    ax.set_title(f"{title} (Curvature: {curvature:.2f})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig