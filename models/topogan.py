import torch
import torch.nn as nn
from ..core.topological_dynamics import TopologicalDynamics

class Generator(nn.Module):
    def __init__(self, latent_dim=128, output_dim=28*28):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.fc(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.topo_dyn = TopologicalDynamics()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(128*7*7, 1)
    
    def forward(self, img):
        img = self.topo_dyn(img)
        feats = self.conv(img)
        feats = feats.view(img.size(0), -1)
        validity = self.fc(feats)
        return validity

class TopoGAN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)
