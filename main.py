import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Dict

from predictive_blocks import PredictiveBlock, PolicyBlock
from world_model import WorldModelGraph

# Assuming PredictiveBlock and WorldModelGraph are already defined as above.

# --- Step 1: Create the graph with a circular dependency (temporal resolution) ---

# Suppose the environment observation space is a vector (like in CartPole)
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]  # Should be 4 for CartPole
env.reset(seed=42)

# Block dimensions:
latent_dim = 32
# Each block will predict next observation dimension as a simplistic target
output_dim = obs_dim

# Create blocks
blockA = PredictiveBlock(input_dim=obs_dim + latent_dim, latent_dim=latent_dim, output_dim=obs_dim)
blockB = PredictiveBlock(input_dim=latent_dim, latent_dim=latent_dim, output_dim=obs_dim)
blockC = PredictiveBlock(input_dim=latent_dim*2, latent_dim=latent_dim, output_dim=obs_dim)

# Build the graph
graph = WorldModelGraph()

# Build the policy block
policy_block = PolicyBlock(latent_dim, env.action_space.n)


# Add the blocks to the graph:
# A depends on C(t-1) and also takes direct env input at each timestep.
# We handle direct env input by leaving a placeholder in the graph and supplying it at runtime.
graph.add_block('A', blockA, inputs=['C'])  # plus external observation at runtime
# B depends on A(t-1)
graph.add_block('B', blockB, inputs=['A'])
# C depends on A(t-1) and B(t-1)
graph.add_block('C', blockC, inputs=['A', 'B'])

# --- Step 2: Simple training loop on environment rollouts ---

optimizer = optim.Adam(graph.blocks['A'].parameters(), lr=1e-3)
optimizer.add_param_group({'params': graph.blocks['B'].parameters()})
optimizer.add_param_group({'params': graph.blocks['C'].parameters()})

loss_fn = nn.MSELoss()

# For this demo, let's just run a single episode and do some training online.
num_episodes = 300
max_steps = 1000

device = torch.device('cpu')
graph.to(device)
policy_block.to(device)

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]

    # We'll store (predictions, targets) for each step and backprop after the episode
    predictions = []
    targets = []

    # We must initialize timestep 0. At t=0, we have no previous latents:
    # Let's initialize outputs[t] dict to empty. We'll set t=-1 to represent no previous step.
    graph.outputs[-1] = {}
    graph.predictions[-1] = {}
    # Initialize latents for A, B, C at t=-1 as zero to break the chain:
    init_latent = torch.zeros((1, latent_dim), device=device)
    graph.outputs[-1]['A'] = init_latent
    graph.outputs[-1]['B'] = init_latent
    graph.outputs[-1]['C'] = init_latent

    for t in range(max_steps):
        # At each timestep, we must prepare inputs:
        inputs_dict = graph.prepare_inputs_for_timestep(t)

        # We know 'A' also needs the current observation.
        # 'A' expects obs_dim + latent_dim inputs: we have 'C' latent from t-1 in inputs_dict,
        # Let's combine with the current observation.
        # inputs_dict['A'] currently is the concatenation of its dependencies at t-1: just 'C'
        # which should be shape [1, latent_dim]. We must concatenate the current observation:
        a_input = torch.cat([obs, inputs_dict['A']], dim=-1)
        inputs_dict['A'] = a_input

        # For 'B', it depends only on 'A(t-1)', which is already in inputs_dict['B']
        # For 'C', depends on 'A(t-1)', 'B(t-1)', concatenation is already handled by prepare_inputs_for_timestep.

        # Now do a forward timestep
        graph.forward_timestep(t, inputs_dict)

        # Get the current prediction of next observation from a chosen block, for example from block A:
        pred_next_obs = graph.get_predictions(t)['A']  # shape [1, obs_dim]
        predictions.append(pred_next_obs)

        # At each timestep t after you have computed your latents:
        latent_A_t = graph.get_outputs(t)['A']  # for example, use block A's latent
        action_logits = policy_block(latent_A_t)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()  # pick an action


        # # Take an action in the environment. Let's do a random policy for demo.
        # action = env.action_space.sample()
        # next_obs, reward, done, truncated, info = env.step(action)
        # next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # The target for the prediction is the next observation:
        targets.append(next_obs_t)

        obs = next_obs_t

        if done or truncated:
            break

    # After the episode, compute the loss for all timesteps and backprop
    optimizer.zero_grad()
    total_loss = 0.0
    for pred, tgt in zip(predictions, targets):
        total_loss += loss_fn(pred, tgt)
    total_loss.backward()
    optimizer.step()

    print(f"Episode {ep}, Steps: {len(predictions)}, Loss: {total_loss.item()}")

env.close()
