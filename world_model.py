import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

from predictive_blocks import OnlinePredictiveBlock


class TwoForwardOneBackBlock(nn.Module):
    def __init__(self, block_factory, lr=1e-3, is_outer=False, token_depth=16):
        super().__init__()
        self.learning_rate = lr

        # Store blocks by name
        self.blocks: Dict[str, OnlinePredictiveBlock] = {}
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


    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.last_pred is not None:
            loss = self.loss_fn(self.last_pred, x_t)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #loss.backward(retain_graph=not self.is_outer)
            self.optimizer.step()

        # 2) Encode x_t into latent representation z_t
        z_t = self.blocks['F1'](x_t)
        z_t_2 = self.blocks['F2'](z_t)
        self.last_latent = torch.cat([z_t, z_t_2], dim=1)
        self.last_pred = self.blocks['B1'](self.last_latent)
        return self.last_latent


class WorldModelGraph:
    """
    Manages a collection of PredictiveBlocks and their dependencies.
    Each block depends on outputs from certain other blocks at the previous timestep.

    We assume all dependencies are from timestep t-1 to t, so no instantaneous cycles.
    """

    def __init__(self):
        # Store blocks by name
        self.blocks: Dict[str, OnlinePredictiveBlock] = {}

        # Dependencies: block_inputs[block_name] = list of block names this block depends on
        self.block_inputs: Dict[str, List[str]] = {}

        # Store the output latents from previous timesteps:
        # outputs[t][block_name] = latent tensor for that block at timestep t
        self.outputs = {}

        # Predicted next inputs: predictions[t][block_name] = predicted input/target at timestep t
        self.predictions = {}

    def add_block(self, name: str, block: OnlinePredictiveBlock, inputs: List[str]):
        self.blocks[name] = block
        self.block_inputs[name] = inputs

    def to(self, device: torch.device):
        for block in self.blocks.values():
            block.to(device)

    def forward_timestep(self, t: int, inputs_dict: Dict[str, torch.Tensor]) -> None:
        """
        Compute one timestep of the graph.

        inputs_dict: Mapping from block name to input vector for that timestep.
                     For blocks that depend on other blocks, the input vector is the
                     concatenation of the latents from those blocks at t-1.

        We assume inputs_dict already contains the concatenated inputs for each block.
        """
        self.outputs[t] = {}
        self.predictions[t] = {}

        for block_name, block in self.blocks.items():
            # Forward pass
            block_input = inputs_dict[block_name]  # [batch_size, input_dim]
            latent = block(block_input)
            self.outputs[t][block_name] = latent

    def prepare_inputs_for_timestep(self, t: int) -> Dict[str, torch.Tensor]:
        """
        Prepares the input vectors for each block at timestep t.
        For each block, we gather the latents from its dependencies at timestep t-1 and concatenate them.

        If t == 0, we assume these are given as initial inputs (e.g. from environment sensors).
        """
        inputs_dict = {}
        for block_name in self.blocks.keys():
            deps = self.block_inputs[block_name]
            if len(deps) == 0:
                # No dependencies, external input required at each timestep (like sensors)
                # The trainer will provide this as external input.
                # We'll initialize with None here and let the trainer overwrite.
                inputs_dict[block_name] = None
            else:
                # Concatenate latents from dependencies at t-1
                dep_latents = [self.blocks[dep_name].last_latent for dep_name in deps]
                inputs_dict[block_name] = torch.cat(dep_latents, dim=1)

        return inputs_dict

    def get_predictions(self, t: int) -> Dict[str, torch.Tensor]:
        return self.predictions[t]

    def get_outputs(self, t: int) -> Dict[str, torch.Tensor]:
        return self.outputs[t]
