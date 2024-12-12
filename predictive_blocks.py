import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict


class PredictiveBlock(nn.Module):
    """
    A simple predictive block that:
    - Takes an input vector (latent from previous timestep)
    - Produces a new latent vector and predicts the next input (or some target)

    Later, you can add attention, memory, or other complexity here.
    """

    def __init__(self, input_dim: int, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Example architecture: input -> MLP -> latent -> MLP -> output
        # You can replace with a transformer-based encoder/decoder here.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Linear(latent_dim, output_dim)  # predict next input or target

    def forward(self, input_vec: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass:
        input_vec: Tensor of shape [batch_size, input_dim]

        Returns:
        - latent_vec: Tensor of shape [batch_size, latent_dim] (latent representation for this timestep)
        - prediction: Tensor of shape [batch_size, output_dim] (prediction of next input)
        """
        latent_vec = self.encoder(input_vec)
        prediction = self.decoder(latent_vec)
        return latent_vec, prediction


class PolicyBlock(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # logits for each action
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Returns logits for each action
        return self.net(latent)


class ValueBlock(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # scalar value
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)
