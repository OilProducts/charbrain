import torch
import torch.nn as nn


# A sensor has to take the observation and shape it into something the attention block can use as
# input.  This will be a vector of (batch, depth_model, len) where depth_model is the depth of
# thought (analogous to token dimensionality and width is the number of "tokens".  It then
# needs to do the opposite on the back.

class LinearToTokenSensor:
    def __init__(self,
                 model_depth:int,
                 observation_width: int,
                 ):
        self.sensor_in = nn.Sequential(
            nn.Conv1d(1, out_channels=model_depth, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.sensor_out = nn.Sequential(
            nn.ConvTranspose1d(model_depth, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Linear(in_features=observation_width, out_features=observation_width)
        )
