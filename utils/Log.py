# pytron_qft/utils/log.py
import logging
import datetime
import json
import torch

class QuantumLogger:
    def __init__(self, log_dir="logs", level=logging.INFO):
        self.log_dir = log_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{log_dir}/quantum_run_{timestamp}.log"
        
        logging.basicConfig(
            filename=self.log_file,
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        
        # Add console output
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        
    def log_metrics(self, metrics):
        """Log training metrics in structured format"""
        logging.info("METRICS: " + json.dumps(metrics))
        
    def log_model(self, model):
        """Log model architecture and parameters"""
        logging.info("MODEL ARCHITECTURE:")
        logging.info(str(model))
        
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"TOTAL PARAMETERS: {param_count}")
        
    def log_wavefunction(self, psi, step=0):
        """Log wavefunction properties"""
        entropy = torch.special.entr(psi.abs().square()).sum()
        logging.info(f"Step {step}: Entropy={entropy.item():.4f}, "
                     f"Max Amplitude={psi.abs().max().item():.4f}")
        
    def log_hardware(self, hardware_info):
        """Log quantum hardware details"""
        logging.info("HARDWARE: " + json.dumps(hardware_info))