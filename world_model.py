import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

from predictive_blocks import AttentionBlock


class TwoForwardOneBackBlock(nn.Module):
    def __init__(self, block_factory, lr=1e-4, is_outer=False, token_depth=16, sensor_adapter=None):
        super().__init__()
        self.learning_rate = lr

        # Store blocks by name
        self.blocks: Dict[str, AttentionBlock] = {}
        self.blocks['F1'] = block_factory()
        self.blocks['F2'] = block_factory()
        self.blocks['B1'] = block_factory()

        self.last_pred = None
        self.last_latent = None
        self.is_outer = is_outer

        self.optimizer = optim.Adam(list(self.blocks['F1'].parameters()) +
                                    list(self.blocks['F2'].parameters()) +
                                    list(self.blocks['B1'].parameters()),
                                    lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def eval_no_backward(self, x_t):
        z_t = self.blocks['F1'](x_t)
        z_t_2 = self.blocks['F2'](z_t)
        return torch.cat([z_t, z_t_2], dim=1).detach()

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.last_pred is not None:
            loss = self.loss_fn(self.last_pred, x_t)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # loss.backward(retain_graph=not self.is_outer)
            self.optimizer.step()

        # 2) Encode x_t into latent representation z_t
        z_t = self.blocks['F1'](x_t)
        z_t_2 = self.blocks['F2'](z_t)
        self.last_latent = torch.cat([z_t, z_t_2], dim=1)
        self.last_pred = self.blocks['B1'](self.last_latent)
        return self.last_latent


class TwoForwardOneBackWithSensor(nn.Module):
    def __init__(self, block_factory, lr=None, is_outer=False, token_depth=16, sensor_adapter=None):
        super().__init__()
        self.learning_rate = lr

        # Store blocks by name
        self.blocks: Dict[str, AttentionBlock] = {'F1': block_factory(), 'F2': block_factory(),
                                                  'B1': block_factory()}

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

    def eval_no_backward(self, x_t):
        if self.is_outer:
            x_t = self.sensor_adapter.sensor_in(x_t).transpose(1, 2)
            z_t = self.blocks['F1'].eval_no_backward(x_t)
            z_t_2 = self.blocks['F2'].eval_no_backward(z_t)
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
