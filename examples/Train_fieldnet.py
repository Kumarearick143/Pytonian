import torch
from pytron_qft.data.mnist import QuantumMNIST
from pytron_qft.models.fieldnet import FieldNet
from pytron_qft.training import QFTrainer, RGFlowCallback
from pytron_qft.hardware import Simulator

def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10
    
    # Data
    qmnist = QuantumMNIST(batch_size=batch_size)
    train_loader, test_loader = qmnist.get_loaders()
    
    # Model
    model = FieldNet(input_dims=[2, 3], hidden_dim=256, num_classes=10)
    
    # Hardware interface
    quantum_hardware = Simulator()
    model.evolution.execute = quantum_hardware.execute_evolution
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    callbacks = [RGFlowCallback(model, frequency=100)]
    trainer = QFTrainer(model, optimizer, device, callbacks=callbacks)
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, accuracy, entropy, betti = trainer.validate(test_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%, Entropy: {entropy:.4f}, "
              f"Betti: {betti}")

if __name__ == "__main__":
    main()