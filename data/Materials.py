import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pymatgen.core import Structure
from .field_ops import spatial_to_spectral, renormalize_field

class QuantumMaterials(Dataset):
    def __init__(self, data_path, structure_column='structure',
                 target_column='formation_energy', max_atoms=100,
                 spectral=True, renormalize=True, cutoff=1e-3):
        """
        Quantum Field Representation of Materials Science Data
        
        Args:
            data_path (str): Path to CSV/JSON file containing material data
            structure_column (str): Column name containing crystal structures
            target_column (str): Column name containing target property
            max_atoms (int): Maximum number of atoms to consider
            spectral (bool): Convert to spectral representation
            renormalize (bool): Apply renormalization cutoff
            cutoff (float): Renormalization cutoff value
        """
        self.spectral = spectral
        self.renormalize = renormalize
        self.cutoff = cutoff
        self.max_atoms = max_atoms
        
        # Load material data
        self.data = pd.read_csv(data_path) if data_path.endswith('.csv') \
            else pd.read_json(data_path)
        
        self.structures = self.data[structure_column]
        self.targets = self.data[target_column].values
        self.element_map = self._create_element_map()
        
    def _create_element_map(self):
        """Create mapping of elements to atomic numbers"""
        all_elements = set()
        for struct_str in self.structures:
            struct = Structure.from_str(struct_str, fmt='json')
            all_elements.update([e.symbol for e in struct.composition.elements])
        
        return {elem: i+1 for i, elem in enumerate(sorted(all_elements))}
    
    def _structure_to_field(self, structure):
        """Convert crystal structure to quantum field representation"""
        # Create 3D grid representation
        grid_size = 32
        density_field = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.cfloat)
        
        # Place atoms in grid
        for site in structure:
            # Get atom type index
            atom_type = self.element_map[site.species_string]
            
            # Calculate grid position
            pos = site.frac_coords
            grid_pos = (pos * grid_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, grid_size-1)
            
            # Add atom to field (complex representation)
            density_field[grid_pos[0], grid_pos[1], grid_pos[2]] += atom_type * (1 + 1j)
        
        return density_field
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Parse structure
        struct_str = self.structures.iloc[idx]
        structure = Structure.from_str(struct_str, fmt='json')
        
        # Create field representation
        field = self._structure_to_field(structure)
        
        # Apply spectral transformation
        if self.spectral:
            field = spatial_to_spectral(field)
            
        # Renormalize small amplitudes
        if self.renormalize:
            field = renormalize_field(field, self.cutoff)
            
        # Get target property
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        
        return field, target

class SyntheticMaterials(Dataset):
    def __init__(self, num_samples=1000, grid_size=32, spectral=True):
        """
        Synthetic Materials Dataset for Quantum Field Representation
        
        Args:
            num_samples (int): Number of samples to generate
            grid_size (int): Size of 3D grid
            spectral (bool): Convert to spectral representation
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.spectral = spectral
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random 3D field (electron density representation)
        real_part = torch.randn(self.grid_size, self.grid_size, self.grid_size)
        imag_part = torch.randn(self.grid_size, self.grid_size, self.grid_size)
        field = torch.complex(real_part, imag_part)
        
        # Apply spectral transformation
        if self.spectral:
            field = spatial_to_spectral(field)
            
        # Create random target property (formation energy)
        target = torch.randn(1).item()
        
        return field, target

def create_materials_loaders(data_path, batch_size=8, spectral=True):
    """
    Create materials science data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    full_dataset = QuantumMaterials(data_path, spectral=spectral)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def create_synthetic_loaders(batch_size=8, spectral=True):
    """
    Create synthetic materials data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    full_dataset = SyntheticMaterials(num_samples=1000, spectral=spectral)
    
    # Split dataset
    train_size = 700
    val_size = 150
    test_size = 150
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader

# Example usage:
# For real data: 
# train_loader, val_loader, test_loader = create_materials_loaders('materials_data.csv')
#
# For synthetic data:
# train_loader, val_loader, test_loader = create_synthetic_loaders()