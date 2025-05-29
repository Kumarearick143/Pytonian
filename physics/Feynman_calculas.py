import torch

class FeynmanPathSimulator:
    def __init__(self, hamiltonian, dt=0.01, n_steps=100):
        """
        Simulate wavefunction evolution by evaluating Feynman path integrals numerically.
        
        Args:
            hamiltonian: callable (psi) -> H psi
            dt: timestep size
            n_steps: number of time steps
        """
        self.hamiltonian = hamiltonian
        self.dt = dt
        self.n_steps = n_steps
    
    def propagate(self, psi0):
        """
        Propagate initial wavefunction psi0 using Trotter approximations of path integrals.
        
        Args:
            psi0: initial wavefunction tensor
        
        Returns:
            psi_t: propagated wavefunction tensor after n_steps
        """
        psi_t = psi0.clone()
        for _ in range(self.n_steps):
            # Apply unitary evolution e^{-i H dt} ≈ 1 - i H dt (first order)
            H_psi = self.hamiltonian(psi_t)
            psi_t = psi_t - 1j * self.dt * H_psi
            psi_t = psi_t / psi_t.norm()  # Normalize wavefunction
        return psi_t
