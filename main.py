import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from typing import Dict, List

from predictive_blocks import PolicyBlock, ValueBlock, AttentionBlock
from world_model import TwoForwardOneBackBlock
from utils import MovingAverage

torch.autograd.set_detect_anomaly(True)

# Assuming PredictiveBlock and WorldModelGraph are already defined as above.

# --- Step 1: Create the graph with a circular dependency (temporal resolution) ---

# Suppose the environment observation space is a vector (like in CartPole)
# env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v3')
obs_dim = env.observation_space.shape[0]  # Should be 4 for CartPole
action_dim = env.action_space.n
env.reset(seed=42)


# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Block dimensions:
latent_dim = 16
depth_of_thought = 16
# Each block will predict next observation dimension as a simplistic target
output_dim = obs_dim
hidden_dim = 32

print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

# Create blocks
# blockA = OnlinePredictiveBlock(input_dim=obs_dim + latent_dim, token_depth=latent_dim,
#                                device=device)
# blockB = OnlinePredictiveBlock(input_dim=latent_dim, token_depth=latent_dim, device=device)
# blockC = OnlinePredictiveBlock(input_dim=latent_dim * 2, token_depth=latent_dim, device=device)


# Build the graph
# graph = WorldModelGraph()

block_factory = lambda: AttentionBlock(d_model=latent_dim, output_length=8, device=device)
model = TwoForwardOneBackBlock(block_factory, lr=1e-5, is_outer=True)
model.to(device)

# Policy and Value blocks:
policy_block = PolicyBlock(latent_dim * 8 * 2, action_dim).to(device)
value_block = ValueBlock(latent_dim * 8 * 2).to(device)

# Sensor block requires input of obs_dim and outputs depth_of_thought
sensor_block = nn.Sequential(
    nn.Conv1d(1, out_channels=depth_of_thought, kernel_size=3, padding=1),
    nn.ReLU(),
).to(device)
# --- Step 2: Simple training loop on environment rollouts ---

# Combine all params in single optimizer for simplicity
optimizer = optim.Adam(list(policy_block.parameters()) +
                       list(value_block.parameters()) +
                       list(sensor_block.parameters()),
                       lr=1e-5)

loss_fn = nn.MSELoss()

# Training loop parameters
num_episodes = 100000
max_steps = 2000
gamma = 0.99

# graph.to(device)
policy_block.to(device)

ma = MovingAverage(window_size=100, mode='simple')

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]

    log_probs = []
    values = []
    rewards: List[torch.Tensor] = []

    for t in range(max_steps):

        enriched_from_sensor = sensor_block(obs.unsqueeze(1)).permute(0, 2, 1)  # reshape to

        out = model.forward(enriched_from_sensor)

        action_logits = policy_block(out.flatten(1).detach())
        value_pred = value_block(out.flatten(1).detach())  # shape [1, 1]

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
    total_reward = sum(rewards).item()
    ma.add_value(total_reward)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r.item() + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)  # [T, 1]

    values = torch.cat(values, dim=0)  # [T, 1]
    log_probs = torch.cat(log_probs, dim=0)  # [T]

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

    print(f"Episode {ep}: Reward={total_reward}, Reward MA={ma}, Return"
          f"={returns.sum().item():.2f}, "
          f"Policy Loss"
          f"={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}")

env.close()
