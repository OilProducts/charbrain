import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict


class PredictiveBlock(nn.Module):
    """
    A predictive block that:
      - Takes in the previous latent representation z_{t-1}.
      - Predicts the current input X_t from z_{t-1}.
      - Immediately computes loss w.r.t. the actual X_t, backpropagates, and updates its parameters.
      - Finally, produces a new latent z_t by encoding the current input X_t.

    This design puts the training step (loss computation + optimizer step) directly inside .forward().
    While unusual, it achieves the "train on every forward call" behavior you requested.
    """

    def __init__(self, latent_dim: int, input_dim: int, hidden_dim: int,
                 optimizer: optim.Optimizer = optim.Adam):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Example architecture:
        #   decode z_{t-1} -> predict X_t
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        #   encode X_t -> produce z_t
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.optimizer = optimizer(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, z_prev: torch.Tensor, x_curr: torch.Tensor):
        """
        z_prev: [batch_size, latent_dim] - latent from the previous timestep
        x_curr: [batch_size, input_dim]  - the current input/observation

        Returns:
          z_curr: new latent representation for this timestep
          x_pred: predicted version of x_curr
          loss:   MSE loss computed between x_pred and x_curr
        """
        # Predict current input from previous latent
        x_pred = self.decoder(z_prev)  # shape: [batch_size, input_dim]

        # Compute loss
        loss = self.loss_fn(x_pred, x_curr)

        # Backpropagate immediately
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Encode the current input to produce the new latent representation
        z_curr = self.encoder(x_curr)  # shape: [batch_size, latent_dim]

        return z_curr, x_pred, loss


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
