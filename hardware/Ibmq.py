# pytron_qft/hardware/ibmq.py
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.ibmq import IBMQ

class IBMQInterface:
    def __init__(self, api_token, backend_name='ibmq_qasm_simulator'):
        self.api_token = api_token
        self.backend_name = backend_name
        self.provider = None
        self.backend = None
        
    def connect(self):
        IBMQ.save_account(self.api_token, overwrite=True)
        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='ibm-q')
        self.backend = self.provider.get_backend(self.backend_name)
        return self.backend.status().operational
    
    def execute_evolution(self, hamiltonian, state, time):
        """Execute Hamiltonian evolution on IBM hardware"""
        n_qubits = len(state) // 2
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # State preparation
        for i, amp in enumerate(state):
            bin_str = format(i, f'0{n_qubits}b')
            qc.initialize({bin_str: amp}, range(n_qubits))
        
        # Hamiltonian evolution
        qc.hamiltonian(hamiltonian, time, range(n_qubits))
        
        # Execute
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Convert to wavefunction
        total_shots = sum(counts.values())
        wavefunction = [0] * (2**n_qubits)
        for state_str, count in counts.items():
            idx = int(state_str, 2)
            wavefunction[idx] = count / total_shots
        
        return torch.tensor(wavefunction, dtype=torch.cfloat)