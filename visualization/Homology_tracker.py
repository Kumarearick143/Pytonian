# pytron_qft/visualization/homology_tracker.py
from gudhi import RipsComplex, plot_persistence_diagram
import matplotlib.pyplot as plt

def track_homology(wavefunction, max_dim=2):
    """Compute and visualize persistent homology"""
    # Convert wavefunction to point cloud
    probs = wavefunction.abs().square().numpy()
    point_cloud = np.vstack([np.arange(len(probs)), probs]).T
    
    # Compute persistent homology
    rips = RipsComplex(points=point_cloud)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    diag = st.persistence()
    
    # Plot persistence diagram
    plt.figure(figsize=(8, 6))
    plot_persistence_diagram(diag)
    plt.title("Persistence Diagram")
    plt.tight_layout()
    
    return diag