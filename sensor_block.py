import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorBlock(nn.Module):
    """

    """
    def __init__(self, input_dim: int,
                 depth_of_thought: int):
        """
        input_dim is the length of the observation vector.
        depth_of_thought is equivalent to the embedding dimension of the latent space.
        latent_dim is the length of the latent vector, equivalent to sequence length.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, out_channels=depth_of_thought, kernel_size=3),
            nn.ReLU(),
        )
