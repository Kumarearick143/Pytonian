import torch.optim.lr_scheduler as lr_scheduler

class QFTScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, anneal_start=10, anneal_end=100, last_epoch=-1):
        """
        Custom temperature schedule that decays temperature from 1.0 to 0 over a range of epochs.
        Useful for annealing-like temperature control in path integral optimizers.
        
        Args:
            optimizer: optimizer to schedule
            anneal_start: epoch where annealing starts
            anneal_end: epoch where annealing ends
        """
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Calculate annealing factor between 1 (start) and 0 (end)
        current_epoch = self.last_epoch
        if current_epoch < self.anneal_start:
            factor = 1.0
        elif current_epoch > self.anneal_end:
            factor = 0.0
        else:
            factor = 1.0 - (current_epoch - self.anneal_start) / (self.anneal_end - self.anneal_start)
        # Apply factor multiplicatively to base learning rates
        return [base_lr * factor for base_lr in self.base_lrs]
