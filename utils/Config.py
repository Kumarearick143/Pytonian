# pytron_qft/utils/config.py
import json
import os
from dataclasses import dataclass

@dataclass
class QuantumConfig:
    # Quantum field parameters
    field_dims: list = (2, 3)
    cutoff_scale: float = 1e-3
    spectral_encoding: bool = True
    
    # Evolution parameters
    time_step: float = 0.1
    hamiltonian_type: str = "learned"  # "fixed" or "learned"
    
    # Measurement parameters
    curvature: float = 0.1
    
    # Hardware parameters
    hardware_backend: str = "simulator"  # "ibmq", "google", "photonic"
    shots: int = 1024
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def hardware_setup(self):
        """Configure hardware based on settings"""
        if self.hardware_backend == "ibmq":
            from Pytonian.hardware import IBMQInterface
            return IBMQInterface(shots=self.shots)
        elif self.hardware_backend == "google":
            from Pytonian.hardware import GoogleQuantumInterface
            return GoogleQuantumInterface()
        elif self.hardware_backend == "photonic":
            from Pytonian.hardware import PhotonicQuantumInterface
            return PhotonicQuantumInterface()
        else:
            from Pytonian.hardware.simulator import Simulator
            return Simulator()

# Default configuration
default_config = QuantumConfig()

def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        return QuantumConfig.load(config_path)
    return default_config