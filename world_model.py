import torch
from typing import List, Dict

from predictive_blocks import OnlinePredictiveBlock


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

        # if t == 0:
        #     # For t=0, no previous outputs exist. We must rely on external input for blocks without dependencies.
        #     # For blocks with dependencies, either initialize them or raise an error.
        #     # Here we’ll assume that if a block has dependencies, you must provide initial state in some way.
        #     # Otherwise, just return an empty vector or rely on a provided initialization.
        #     for block_name in self.blocks.keys():
        #         # For simplicity, we expect the trainer to provide initial inputs for t=0.
        #         # We'll just put a placeholder here:
        #         raise RuntimeError("Initial inputs for timestep 0 must be provided externally.")
        # else:
        for block_name in self.blocks.keys():
            deps = self.block_inputs[block_name]
            if len(deps) == 0:
                # No dependencies, external input required at each timestep (like sensors)
                # The trainer will provide this as external input.
                # We'll initialize with None here and let the trainer overwrite.
                inputs_dict[block_name] = None
            else:
                # Concatenate latents from dependencies at t-1
                dep_latents = [self.outputs[t - 1][dep_name] for dep_name in deps]
                inputs_dict[block_name] = torch.cat(dep_latents, dim=-1)

        return inputs_dict

    def get_predictions(self, t: int) -> Dict[str, torch.Tensor]:
        return self.predictions[t]

    def get_outputs(self, t: int) -> Dict[str, torch.Tensor]:
        return self.outputs[t]
