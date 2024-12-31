import cProfile
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import Dict, List

from sensor_block import LinearToTokenSensor
from predictive_blocks import TwoLayerLinearBlock, AttentionBlock
from world_model import TwoForwardOneBackWithSensor, TwoForwardOneBackBlock, CircularWorld
from utils import MovingAverage

# torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')


# env = gym.make('CartPole-v1')
num_envs = 2
env_name = "LunarLander-v3"
envs = gym.make_vec(env_name, num_envs)
obs_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Block dimensions:
sensor_width = obs_dim
depth_of_thought = 16
width_of_thought = 8
# Each block will predict next observation dimension as a simplistic target
output_dim = obs_dim
# hidden_dim = 32

print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

# Adapts from (Batch, 8) to (Batch, depth_of_thought, 8).  8 is the first width
sensor_adapter = LinearToTokenSensor(model_depth=depth_of_thought, observation_width=obs_dim)

block_factory = lambda **kwargs: AttentionBlock(device=device,
                                                depth_model=depth_of_thought,
                                                **kwargs)


sensor_model = TwoForwardOneBackWithSensor(block_factory, lr=1e-5,
                                           is_outer=True,
                                           sensor_adapter=sensor_adapter)

world_block_factory = lambda input_merge_width: TwoForwardOneBackBlock(block_factory,
                                                     depth_model=depth_of_thought,
                                                     input_width=32,
                                                     input_merge_width=input_merge_width,
                                                     lr=1e-5,
                                                     is_outer=False)
world_model = CircularWorld(block_factory=world_block_factory, width=16, onboard_width=16)

world_model.to(device)

# Policy and Value blocks:
# Depth of thought * width of thought * 2 * 2 because we concatenate two latents from world
# model, and two latents from the sensor model
policy_block = TwoLayerLinearBlock((depth_of_thought * 80),
                                   action_dim, hidden_dim=1024).to(
    device)
policy_optimizer = optim.Adam(policy_block.parameters(), lr=1e-5)

value_block = TwoLayerLinearBlock((depth_of_thought * 80), 1,
                                  hidden_dim=1024).to(device)
value_optimizer = optim.Adam(value_block.parameters(), lr=1e-5)

# Combine all params in single optimizer for simplicity
# optimizer = optim.Adam(list(policy_block.parameters()) +
#                        lr=1e-5)

loss_fn = nn.MSELoss()

# Training loop parameters
max_episodes = 5000
max_steps = 2000000
gamma = 0.99


def make_env():
    return gym.make("LunarLander-v3")


def train():
    env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    obs, info = env.reset()

    obs_t = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(1)
    episode_rewards = np.zeros(num_envs)

    ma = MovingAverage(window_size=100, mode='simple')
    episodes_completed = 0
    for step_count in range(max_steps):
        # 1) World model forward
        input_to_model = obs_t  # .transpose(1, 2)  # (num_envs, input_dim, seq_len)
        sensor_latent = sensor_model(input_to_model)  # (batch, depth_of_thought, width_of_thought)
        world_latent = world_model(sensor_latent)  # (batch, depth_of_thought, width_of_thought)
        latent = torch.cat([sensor_latent, world_latent],
                           dim=1)  # (batch, depth_of_thought, width_of_thought * 2)
        latent_flat = latent.reshape(latent.shape[0], -1).detach()  # (num_envs, 256)

        # 2) Policy forward
        logits = policy_block(latent_flat)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        # 3) Step environment
        next_obs, rewards, dones, truncated, infos = env.step(actions.cpu().numpy())
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device).unsqueeze(1)

        # 4) Compute value estimates (for actor-critic)
        values = value_block(latent_flat)  # V(current_state)

        # For the next state, need to get its latent too:
        with torch.no_grad():
            next_input_to_model = next_obs_t  # .transpose(1, 2)
            next_sensor_latent = sensor_model.eval_no_grad(next_input_to_model).detach()
            next_world_latent = world_model.eval_no_grad(next_sensor_latent).detach()
            next_latent = torch.cat([next_sensor_latent, next_world_latent], dim=1)
            next_latent_flat = next_latent.reshape(next_latent.shape[0], -1)
            next_values = value_block(next_latent_flat)  # V(next_state), no grad for now

        # 5) Compute advantages using TD(0):
        # advantage = R + gamma * V(next_state) - V(state)
        # If done, V(next_state)=0
        rewards_t = torch.tensor(rewards, device=device)
        mask = torch.tensor(~dones, dtype=torch.float32, device=device)
        targets = torch.tensor(rewards_t, dtype=torch.float32) + gamma * next_values * mask
        advantages = targets - values

        # 6) Update policy
        # Policy loss = -log_prob(a) * advantage
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # 7) Update value network with MSE on the advantage
        # Actually we use advantage as target - V(s), so MSE on (V - target)
        value_loss = (values - targets.detach()).pow(2).mean()

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        policy_optimizer.step()
        value_optimizer.step()

        # Update observation
        obs_t = next_obs_t

        # Track episode rewards
        episode_rewards += rewards
        for i, done_ in enumerate(dones):
            if done_:
                episodes_completed += 1
                ma.add_value(episode_rewards[i])
                if episodes_completed % 10 == 0:
                    print(f"Episode: {episodes_completed}, Reward MA: {ma}")
                episode_rewards[i] = 0.0  # reset for next
                if episodes_completed >= max_episodes:
                    break

        # Optional: print progress
        if step_count % 100 == 0:
            print(f"Step {step_count}, Policy Loss: {policy_loss.item():.4f}, Value Loss: "
                  f"{value_loss.item():.4f}, Sensor Model Loss: {sensor_model.loss:.4f}")




train()

# # Profile the training function
# cProfile.run('train()', 'profile_output')
#
# # Print profiling results
# with open('profile_results.txt', 'w') as f:
#     p = pstats.Stats('profile_output', stream=f)
#     p.sort_stats('cumulative').print_stats(50)
