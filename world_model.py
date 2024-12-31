import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

from predictive_blocks import AttentionBlock


# The block factory output must take at minimum output_width, and depth_model.  Input width is
# used in these classes to ensure the correct shape for the output of B1 so that inputs to the
# loss function will match.

# Three classes are defined here, TwoForwardOneBack is the base case, WithMerge is used to enable
# recurrence, WithSensor is used as a special kind of merge that can onboard new observations.

class TwoForwardOneBackBlock(nn.Module):
    def __init__(self,
                 block_factory,
                 lr=1e-5,
                 is_outer=False,
                 input_merge_width: int = 0,
                 depth_model: int = 16,
                 input_width: int = 8,
                 output_width: int = 16):
        super().__init__()
        self.learning_rate = lr
        self.depth_model = depth_model
        self.input_width = input_width
        self.output_width = output_width

        # Store blocks by name
        self.blocks: Dict[str, AttentionBlock] = {}
        self.blocks['F1'] = block_factory(output_width=input_width)
        self.blocks['F2'] = block_factory(output_width=input_width)
        self.blocks['B1'] = block_factory(output_width=(2 * input_width) + input_merge_width)

        self.last_pred = None
        self.last_latent = None
        self.is_outer = is_outer

        self.optimizer = optim.Adam(list(self.blocks['F1'].parameters()) +
                                    list(self.blocks['F2'].parameters()) +
                                    list(self.blocks['B1'].parameters()),
                                    lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def eval_no_grad(self, x_t):
        z_t = self.blocks['F1'](x_t)
        z_t_2 = self.blocks['F2'](z_t)
        return torch.cat([z_t, z_t_2], dim=1).detach()

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.last_pred is not None:
            loss = self.loss_fn(self.last_pred, x_t)
            self.optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)
            # loss.backward(retain_graph=not self.is_outer)
            self.optimizer.step()

        # 2) Encode x_t into latent representation z_t
        z_t = self.blocks['F1'](x_t)
        z_t_2 = self.blocks['F2'](z_t)
        self.last_latent = torch.cat([z_t, z_t_2], dim=1)
        self.last_pred = self.blocks['B1'](self.last_latent)
        return self.last_latent.detach()


class TwoForwardOneBackWithSensor(nn.Module):
    def __init__(self,
                 block_factory,
                 lr=1e-5,
                 is_outer=False,
                 input_merge_width: int = 0,
                 depth_model: int = 16,
                 input_width: int = 8,
                 output_width: int = 16,
                 sensor_adapter=None):
        super().__init__()
        self.learning_rate = lr

        # Store blocks by name
        self.blocks: Dict[str, AttentionBlock] = {}
        self.blocks['F1'] = block_factory(output_width=input_width)
        self.blocks['F2'] = block_factory(output_width=input_width)
        self.blocks['B1'] = block_factory(output_width=input_width + input_merge_width)

        self.last_pred = None
        self.last_latent = None
        self.is_outer = is_outer
        self.sensor_adapter = sensor_adapter

        self.optimizer = optim.Adam(list(self.blocks['F1'].parameters()) +
                                    list(self.blocks['F2'].parameters()) +
                                    list(self.blocks['B1'].parameters()) +
                                    list(self.sensor_adapter.sensor_in.parameters()) +
                                    list(self.sensor_adapter.sensor_out.parameters()),
                                    lr=self.learning_rate)

        self.loss_fn = nn.MSELoss()
        self.loss = 0

    def eval_no_grad(self, x_t):
        with torch.no_grad():
            x_t = self.sensor_adapter.sensor_in(x_t).transpose(1, 2)
            z_t = self.blocks['F1'].eval_no_grad(x_t)
            z_t_2 = self.blocks['F2'].eval_no_grad(z_t)
            return torch.cat([z_t, z_t_2], dim=1).detach()

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.last_pred is not None:
            loss = self.loss_fn(self.last_pred, x_t)
            self.loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)
            self.optimizer.step()

        # 2) Encode x_t into latent representation z_t
        x_t = self.sensor_adapter.sensor_in(x_t).transpose(1, 2)
        z_t = self.blocks['F1'](x_t)
        z_t_2 = self.blocks['F2'](z_t)
        self.last_latent = torch.cat([z_t, z_t_2], dim=1)
        pred = self.blocks['B1'](self.last_latent).transpose(1, 2)
        self.last_pred = self.sensor_adapter.sensor_out(pred)

        return self.last_latent.detach()


class CircularWorld(nn.Module):
    def __init__(self,
                 block_factory: callable,
                 width: int,
                 onboard_width: int,
                 length: int = 4):
        super().__init__()
        self.blocks = []
        # The first block is special. It's self.last_pred will be (2 * input width) + merge_width
        self.blocks.append(block_factory(input_merge_width=onboard_width))
        for _ in range(length - 1):
            self.blocks.append(block_factory(input_merge_width=0))
        self.last_end = torch.zeros(2, 64, 16)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = self.blocks[0](torch.cat([x_t, self.last_end], dim=1))
        for block in self.blocks[1:]:
            x_t = block(x_t)
        self.last_end = x_t.detach()
        return self.last_end

    def eval_no_grad(self, x_t):
        for block in self.blocks:
            x_t = block.eval_no_grad(x_t)
        return x_t.detach()
