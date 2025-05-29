# examples/train_quantumgpt.py
import torch
import os
from pytron_qft.models import QuantumGPT
from pytron_qft.data import TextDataset
from pytron_qft.training import QGPTrainer
from pytron_qft.utils.config import load_config
from pytron_qft.utils.log import QuantumLogger

def main():
    # Load configuration
    config = load_config("configs/qgpt_config.json")
    logger = QuantumLogger(config.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    dataset = TextDataset(file_path=config.data_path, 
                         block_size=config.block_size)
    loader = torch.utils.data.DataLoader(dataset, 
                                       batch_size=config.batch_size, 
                                       shuffle=True)
    
    # Initialize model
    model = QuantumGPT(
        vocab_size=dataset.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        curvature=config.curvature,
        entanglement_lambda=config.entanglement_lambda
    ).to(device)
    
    # Load checkpoint if exists
    if config.resume_checkpoint:
        model.load_state_dict(torch.load(config.resume_checkpoint))
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Trainer
    trainer = QGPTrainer(model, optimizer, device, config, logger)
    
    # Training loop
    for epoch in range(config.start_epoch, config.epochs):
        # Train one epoch
        train_loss = trainer.train_epoch(loader, epoch)
        
        # Validate
        val_loss = trainer.validate(loader)
        
        # Log metrics
        logger.log_metrics({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": torch.exp(torch.tensor(val_loss)).item()
        })
        
        # Generate sample
        if epoch % config.sample_interval == 0:
            prompt = "The quantum nature of reality"
            generated = model.generate(prompt, length=200)
            logger.log_info(f"Generated text:\n{generated}")
            
            # Save sample
            with open(os.path.join(config.output_dir, f"sample_epoch_{epoch}.txt"), "w") as f:
                f.write(generated)
        
        # Save checkpoint
        if epoch % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"qgpt_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
        
        # Update learning rate
        trainer.update_lr(epoch)

    logger.log_info("Training completed successfully")

if __name__ == "__main__":
    main()