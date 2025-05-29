import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.quantum_field import QuantumField

class QuantumBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.cf = QuantumField(field_dims=[2])
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.cf(x)
        out = self.conv1(out.real.unsqueeze(1))  # Use real part for conv
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x.real.unsqueeze(1))
        out = F.relu(out)
        return out

class QFResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.layer1 = QuantumBasicBlock(1, 64)
        self.layer2 = QuantumBasicBlock(64, 128, stride=2)
        self.layer3 = QuantumBasicBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
