# pytron_qft/training/callbacks.py

class RGFlowCallback:
    """Renormalization Group Flow Callback"""
    def __init__(self, model, frequency=100):
        self.model = model
        self.frequency = frequency
        self.step_count = 0
        
    def on_batch_end(self):
        self.step_count += 1
        if self.step_count % self.frequency == 0:
            self.model.apply_renormalization()

class TopologyLogger:
    """Log topological features during training"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.betti_history = []
        
    def on_epoch_end(self, betti_numbers):
        self.betti_history.append(betti_numbers)
        torch.save(self.betti_history, self.log_path)