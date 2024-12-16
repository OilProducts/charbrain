import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorBlock(nn.Module):
    """

    """
    def __init__(self, input_dim: int,
                 latent_dim: int,
                 depth_of_thought: int, latent_dim: int, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, depth_of_thought)
        )