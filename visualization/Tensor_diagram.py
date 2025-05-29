# pytron_qft/visualization/tensor_diagram.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_tensor_network(tensors, labels=None):
    """Visualize tensor network structure"""
    G = nx.Graph()
    
    # Add nodes
    for i, tensor in enumerate(tensors):
        G.add_node(i, size=np.prod(tensor.shape))
    
    # Add edges based on shared dimensions
    for i in range(len(tensors)):
        for j in range(i+1, len(tensors)):
            shared_dims = set(tensors[i].shape) & set(tensors[j].shape)
            if shared_dims:
                G.add_edge(i, j, weight=len(shared_dims))
    
    # Draw graph
    pos = nx.spring_layout(G)
    node_sizes = [d['size']*100 for _, d in G.nodes(data=True)]
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=[d['weight'] for _, _, d in G.edges(data=True)])
    
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.title("Tensor Network Diagram")
    plt.axis('off')
    return plt.gcf()