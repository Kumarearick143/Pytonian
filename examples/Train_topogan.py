# examples/train_topogan.py
import torch
import os
from pytron_qft.models import TopoGAN
from pytron_qft.data import MaterialDataset
from pytron_qft.training import GANTrainer
from pytron_qft.utils.config import load_config
from pytron_qft.utils.log import QuantumLogger
from pytron_qft.visualization import plot_generated_samples

def main():
    # Load configuration
    config = load_config("configs/topogan_config.json")
    logger = QuantumLogger(config.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    dataset = MaterialDataset(root=config.data_path, 
                             transform=config.transform,
                             download=True)
    loader = torch.utils.data.DataLoader(dataset, 
                                       batch_size=config.batch_size, 
                                       shuffle=True)
    
    # Initialize model
    model = TopoGAN(
        latent_dim=config.latent_dim,
        img_size=config.img_size,
        channels=config.channels,
        curvature=config.curvature,
        topological_lambda=config.topological_lambda
    ).to(device)
    
    # Load checkpoint if exists
    if config.resume_checkpoint:
        model.load_state_dict(torch.load(config.resume_checkpoint))
    
    # Setup optimizers
    opt_g = torch.optim.Adam(model.generator.parameters(), 
                            lr=config.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), 
                            lr=config.lr_d, betas=(0.5, 0.999))
    
    # Trainer
    trainer = GANTrainer(model, opt_g, opt_d, device, config, logger)
    
    # Fixed latent for sample generation
    fixed_z = torch.randn(64, config.latent_dim, device=device)
    
    # Training loop
    for epoch in range(config.start_epoch, config.epochs):
        # Train one epoch
        metrics = trainer.train_epoch(loader, epoch)
        
        # Log metrics
        logger.log_metrics({"epoch": epoch, **metrics})
        
        # Generate samples
        if epoch % config.sample_interval == 0:
            samples = model.generate(fixed_z)
            plot_generated_samples(samples, epoch, config.sample_dir)
        
        # Save checkpoint
        if epoch % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"topogan_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
        
        # Update learning rate
        trainer.update_lr(epoch)

    logger.log_info("Training completed successfully")

if __name__ == "__main__":
    main()