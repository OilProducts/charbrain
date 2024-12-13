import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from typing import Dict

from predictive_blocks import PredictiveBlock, PolicyBlock, ValueBlock
from world_model import WorldModelGraph

# Assuming PredictiveBlock and WorldModelGraph are already defined as above.

# --- Step 1: Create the graph with a circular dependency (temporal resolution) ---

# Suppose the environment observation space is a vector (like in CartPole)
# env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v3')
obs_dim = env.observation_space.shape[0]  # Should be 4 for CartPole
action_dim = env.action_space.n
env.reset(seed=42)

device = torch.device('cpu')


# Block dimensions:
latent_dim = 16
# Each block will predict next observation dimension as a simplistic target
output_dim = obs_dim
hidden_dim = 32

# Create blocks
blockA = PredictiveBlock(input_dim=obs_dim + latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
blockB = PredictiveBlock(input_dim=latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
blockC = PredictiveBlock(input_dim=latent_dim*2, latent_dim=latent_dim, hidden_dim=hidden_dim)

# Build the graph
graph = WorldModelGraph()

# Add the blocks to the graph:
# A depends on C(t-1) and also takes direct env input at each timestep.
# We handle direct env input by leaving a placeholder in the graph and supplying it at runtime.
graph.add_block('A', blockA, inputs=['C'])  # plus external observation at runtime
# B depends on A(t-1)
graph.add_block('B', blockB, inputs=['A'])
# C depends on A(t-1) and B(t-1)
graph.add_block('C', blockC, inputs=['A', 'B'])

# Policy and Value blocks:
policy_block = PolicyBlock(latent_dim, action_dim).to(device)
value_block = ValueBlock(latent_dim).to(device)

# --- Step 2: Simple training loop on environment rollouts ---

# Combine all params in single optimizer for simplicity
optimizer = optim.Adam(list(graph.blocks['A'].parameters()) +
                       list(graph.blocks['B'].parameters()) +
                       list(graph.blocks['C'].parameters()) +
                       list(policy_block.parameters()) +
                       list(value_block.parameters()),
                       lr=1e-4)

loss_fn = nn.MSELoss()

# Training loop parameters
num_episodes = 1000
max_steps = 2000
gamma = 0.99

graph.to(device)
policy_block.to(device)

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]

    # # We'll store (predictions, targets) for each step and backprop after the episode
    # predictions = []
    # targets = []

    # We must initialize timestep 0. At t=0, we have no previous latents:
    # Let's initialize outputs[t] dict to empty. We'll set t=-1 to represent no previous step.
    graph.outputs[-1] = {}
    graph.predictions[-1] = {}
    # Initialize latents for A, B, C at t=-1 as zero to break the chain:
    init_latent = torch.zeros((1, latent_dim), device=device)
    graph.outputs[-1]['A'] = init_latent
    graph.outputs[-1]['B'] = init_latent
    graph.outputs[-1]['C'] = init_latent

    log_probs = []
    values = []
    rewards = []

    for t in range(max_steps):
        # Prepare inputs
        inputs_dict = graph.prepare_inputs_for_timestep(t)
        # For A: concatenate obs_t with C_{t-1}
        a_input = torch.cat([obs, inputs_dict['A']], dim=-1)
        inputs_dict['A'] = a_input

        # Forward timestep in the graph
        graph.forward_timestep(t, inputs_dict)

        # Use block A's latent as state representation for policy and value
        latent_A_t = graph.get_outputs(t)['A']  # shape [1, latent_dim]

        # Compute policy and value
        action_logits = policy_block(latent_A_t)
        value_pred = value_block(latent_A_t)  # shape [1, 1]

        # Sample action
        dist = Categorical(logits=action_logits)
        action = dist.sample()  # shape [1]

        # Step env
        next_obs, reward, done, truncated, info = env.step(action.item())
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Store log_prob, value, reward
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        values.append(value_pred)
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))

        obs = next_obs_t

        if done or truncated:
            break

    # Compute returns and advantages
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r.item() + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1) # [T, 1]

    values = torch.cat(values, dim=0) # [T, 1]
    log_probs = torch.cat(log_probs, dim=0) # [T]

    # Advantage = returns - values
    advantages = returns - values.detach()

    # Policy loss: -log_prob * advantage
    policy_loss = -(log_probs * advantages).mean()

    # Value loss: MSE(returns, values)
    value_loss = (values - returns).pow(2).mean()

    # Optional: If you still want to encourage good prediction performance,
    # you could also add a prediction loss using graph.get_predictions at each timestep.
    # For now, we focus on RL signals only.

    optimizer.zero_grad()
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    optimizer.step()

    print(f"Episode {ep}: Return={returns.sum().item():.2f}, Policy Loss={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}")


env.close()
