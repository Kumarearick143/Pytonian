import torch

class SymmetryDetector:
    def __init__(self, threshold=1e-3):
        """
        Detect symmetries like U(1), SU(n), etc. in operator matrices.
        
        Args:
            threshold: float numerical tolerance for symmetry checking
        """
        self.threshold = threshold
    
    def is_unitary(self, matrix):
        """Check if a matrix is unitary: U†U = I"""
        I = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
        unitary_test = matrix.conj().transpose(-2, -1) @ matrix
        deviation = torch.norm(unitary_test - I)
        return deviation < self.threshold
    
    def detectsymmetry(self, matrix):
        """
        Detect common symmetries. Currently supports unitary detection.
        Extendable to SU(n), U(1), etc.
        
        Returns: str symmetry group name or None
        """
        if self.is_unitary(matrix):
            return "U(n)"  # Placeholder for general unitary group
        # Other symmetry detections can be implemented here
        
        return None
