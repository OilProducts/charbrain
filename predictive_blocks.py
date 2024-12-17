import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict


class OnlinePredictiveBlock(nn.Module):
    """
    A predictive block that:
      - At each forward pass, returns a latent representation for the current input x_t.
      - Internally, it still *predicts* x_{t+1} for the purpose of computing an auto-prediction loss,
        but it does not return that prediction.
      - The next time forward(x_t+1) is called, it backpropagates based on the mismatch between
        the *previous* prediction (stored in self.last_pred) and the *current* ground truth input x_t+1.
      - End result: you get a trained latent representation each step without externally passing around
        the "next input" prediction.

    forward(x_t):
      1) If self.last_pred is not None, compute MSE(last_pred, x_t) -> backprop + step.
      2) Encode x_t -> latent z_t
      3) Produce a new prediction of x_{t+1}, store it as self.last_pred (internally).
      4) Return z_t (the block’s latent representation).
    """

    def __init__(self, input_dim, depth_of_thought, learning_rate=1e-3, device='cpu'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.depth_of_thought = depth_of_thought
        self.output_dim = input_dim

        # Example architecture: simple MLP
        # encoder to produce latent, decoder to produce the next input prediction
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=self.depth_of_thought,
                                                       nhead=1,
                                                       bias=False,
                                                       dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=2)

        self.decoder = nn.TransformerDecoderLayer(d_model=self.depth_of_thought,
                                                  nhead=1,
                                                  bias=False,
                                                  dim_feedforward=64)



        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Linear(latent_dim, self.output_dim)

        # Maintain the last prediction of x_t (used for training at the next forward call)
        self.last_pred = None

        # Internal optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()


def forward(self, x_t: torch.Tensor) -> torch.Tensor:
    """
    x_t: shape [batch_size, input_dim], the current ground-truth input.

    Returns:
        z_t: shape [batch_size, latent_dim], the latent representation for the current input.

    Internally:
        - If self.last_pred is not None, compute loss = MSE(self.last_pred, x_t)
          and do one optimizer step.
        - Then encode x_t into latent z_t.
        - Also produce a new "next input" prediction x_pred_tplus1 = decoder(z_t),
          store it as self.last_pred for next time's training.
    """
    # 1) Train on the previous step's prediction if available
    if self.last_pred is not None:
        loss = self.loss_fn(self.last_pred, x_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 2) Encode x_t into latent representation z_t
    z_t = self.encoder(x_t)

    # # 3) Predict next input from z_t (for the next step's training)
    # x_pred_tplus1 = self.decoder(z_t)
    # # Detach so we don't accumulate a huge graph over many timesteps
    # self.last_pred = x_pred_tplus1.detach()

    self.last_pred = self.decoder(z_t)

    # 4) Return the latent representation
    return z_t.detach()


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
            nn.Linear(64, 1)  # scalar value
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


"""
The class should take a factory that produces the blocks used in this level of the graph.

It should create one that will handle input, its latent should be fed into the second.  Its latent, 
along with the latent of the first should be fed into the reconstruction unit.  
"""
