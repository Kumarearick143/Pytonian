# pytron_qft/hardware/simulator.py
import torch
from abc import ABC, abstractmethod

class QuantumHardwareInterface(ABC):
    @abstractmethod
    def execute_evolution(self, hamiltonian, state, time):
        pass

class Simulator(QuantumHardwareInterface):
    def __init__(self, precision=torch.cfloat):
        self.precision = precision
        
    def execute_evolution(self, hamiltonian, state, time):
        # Unitary evolution: U = exp(-iHt)
        evolution_op = torch.matrix_exp(-1j * hamiltonian * time)
        return torch.matmul(evolution_op, state)
    
    def measure(self, state, observable):
        # Compute expectation value: <ψ|O|ψ>
        return torch.vdot(state, torch.matmul(observable, state)).real