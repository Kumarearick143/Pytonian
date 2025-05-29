# pytron_qft/training/trainer.py
import torch
from tqdm import tqdm
from .metrics import entanglement_entropy, betti_numbers

class QFTrainer:
    def __init__(self, model, optimizer, device, callbacks=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks or []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            data, target = batch
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Quantum evolution
            output = self.model(data)
            
            # Measurement collapse
            loss = self.model.collapse(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Execute callbacks
            for callback in self.callbacks:
                callback.on_batch_end()
        
        return total_loss / len(train_loader)

    def validate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Validation"):
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = self.model.collapse(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                # Compute quantum metrics
                entropy = entanglement_entropy(output)
                betti = betti_numbers(output)
                
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        return avg_loss, accuracy, entropy, betti