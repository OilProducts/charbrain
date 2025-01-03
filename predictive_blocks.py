import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict


class AttentionBlock(nn.Module):
    def __init__(self,
                 # input_width: int,
                 output_width: int = 8,
                 depth_model: int = 16,
                 n_heads: int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()

        # Learned queries for attention-based reduction
        self.query_embed = nn.Parameter(torch.randn(output_width, depth_model)).to(device)

        # Multihead Attention
        self.mha = nn.MultiheadAttention(embed_dim=depth_model, num_heads=n_heads,
                                         batch_first=True).to(
            device)

        # Add normalization
        self.norm1 = nn.LayerNorm(depth_model).to(device)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(depth_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, depth_model),
            nn.ReLU()
        ).to(device)

        self.norm2 = nn.LayerNorm(depth_model).to(device)

        # Optional dropout
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):  # x: (batch, depth_model, input_length)
        # x: (batch, input_width, depth_model)
        batch_size = x.size(0)

        # Repeat queries for each batch
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # Multihead Attention
        attended, _ = self.mha(queries, x, x)
        # attended = self.dropout(attended)
        attended = self.norm1(attended)  # Normalize after attention

        # Feed-forward
        ff_output = self.ffn(attended)
        ff_output = self.dropout(ff_output)

        # Add & Norm
        output = self.norm2(attended + ff_output)

        # output: (batch, output_length, d_model)
        return output

    def eval_no_grad(self, x):
        with torch.no_grad():
            return self(x)


class OnlinePredictiveBlock(nn.Module):
    def __init__(self, input_dim, token_depth, learning_rate=1e-3, device='cpu'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.token_depth = token_depth
        self.output_dim = input_dim

        # Example architecture: simple MLP
        # encoder to produce latent, decoder to produce the next input prediction
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=self.token_depth,
                                                       nhead=1,
                                                       bias=False,
                                                       dim_feedforward=self.token_depth,
                                                       batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=2)

        self.decoder = nn.TransformerEncoderLayer(d_model=self.token_depth,
                                                  nhead=1,
                                                  bias=False,
                                                  dim_feedforward=self.token_depth,
                                                  batch_first=True)

        # Maintain the last prediction of x_t (used for training at the next forward call)
        self.last_pred = None
        self.last_latent = None

        # Internal optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        # 1) Train on the previous step's prediction if available
        if self.last_pred is not None:
            loss = self.loss_fn(self.last_pred, x_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 2) Encode x_t into latent representation z_t
        z_t = self.encoder(x_t)

        # # 3) Predict next input from z_t (for the next step's training)
        self.last_pred = self.decoder(z_t)
        self.last_latent = z_t

        # 4) Return the latent representation
        return self.last_latent


class TwoLayerLinearBlock(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)

        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)
