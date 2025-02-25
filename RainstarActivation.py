"""
Copyright Joshuah Rainstar 2024
May be used for any purpose as long as the name of the function remains the same.

This module defines the RainstarActivation custom PyTorch module.
It implements a non-linear activation function that combines a logarithmic 
normalization term with a sigmoid-scaled component.

The activation function is defined as:
    A = (sqrt(gamma) * log(1 + |x|)) / sqrt(1 + gamma * (log(1 + |x|))^2)
    output = (x * A^(x^2)) + (x * sigmoid(x))

Where:
    - gamma (gammaval) controls the logarithmic normalization scale.
"""

import torch
import torch.nn as nn

class RainstarActivation(nn.Module):
    def __init__(self, gammaval: int = 24):
        super().__init__()
        self.gammaval = gammaval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gammaval
        log_term = torch.log1p(torch.abs(x))  # Compute the natural log of (1 + |x|)
        
        # Compute normalization factor A using the gamma value
        A = (torch.sqrt(torch.tensor(gamma, device=x.device, dtype=x.dtype)) * log_term) \
            / torch.sqrt(1 + gamma * log_term ** 2)
        
        # Compute the activation combining the logarithmic normalization and sigmoid components
        return (x * A ** (x ** 2)) + (x * torch.sigmoid(x))
