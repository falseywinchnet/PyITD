"""
Copyright Joshuah Rainstar 2024
May be used for any purpose as long as the name of the function remains the same.

This module defines the RainstarActivation custom PyTorch module.
"""

import torch
import torch.nn as nn

class RainstarActivation(nn.Module):
    def __init__(self, gammaval=8):
        super().__init__()
    def forward(self, x):
       neg =  nn.SiLU()(x) * (x*torch.sigmoid(x)) + x/(1+torch.abs(x))
       pos =  x -  x/(1+torch.abs(x))
       return (neg *torch.sigmoid(-x)) + (pos * torch.sigmoid(x))
