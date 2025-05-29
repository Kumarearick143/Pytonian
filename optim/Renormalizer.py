import torch
import torch.nn as nn

class Renormalizer(nn.Module):
    def __init__(self, learning_rate=1e-3, momentum=0.9):
        """
        Gradient flow engine using renormalization group inspired smoothing.
        Allows gradient updates with a scale cutoff and flow dynamics.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def forward(self, model, loss):
        """
        Update model's parameters via RG-based gradient flow.
        
        Args:
            model: nn.Module whose parameters we update
            loss: scalar tensor loss to backprop
        
        Returns:
            Updated parameters in-place
        """
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                if self.velocity is None:
                    self.velocity = torch.zeros_like(param.grad)
                self.velocity = self.momentum * self.velocity + self.learning_rate * param.grad
                param.data -= self.velocity
                param.grad.zero_()
        return model
