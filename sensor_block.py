import torch
import torch.nn as nn


# A sensor has to take the observation and shape it into something the attention block can use as input.  This will be a
# vector of (batch, d_model, len) where d_model is the depth of thought and len is the number of "tokens".  It then
# needs to do the opposite on the back.

class SensorAdapter:
    def __init__(self, sensor_in: nn.Module, sensor_out: nn.Module):
        self.sensor_in = sensor_in
        self.sensor_out = sensor_out


class SensorAdapterIn(nn.Module):

    def __init__(self, input_dim: int,
                 depth_of_thought: int, device=torch.device('cpu')):
        """
        input_dim is the length of the observation vector.
        depth_of_thought is equivalent to the embedding dimension of the latent space.
        latent_dim is the length of the latent vector, equivalent to sequence length.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, out_channels=depth_of_thought, kernel_size=3, padding=1),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        return self.net(x)


class SensorAdapterOut(nn.Module):
    def __init__(self, output_dim: int,
                 depth_of_thought: int,
                 device=torch.device('cpu')):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(depth_of_thought, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        return self.net(x)
