import cProfile
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import Dict, List

from sensor_block import SensorAdapter, SensorAdapterIn, SensorAdapterOut
from predictive_blocks import PolicyBlock, ValueBlock, AttentionBlock
from world_model import TwoForwardOneBackWithSensor, TwoForwardOneBackBlock
from utils import MovingAverage

torch.set_float32_matmul_precision('high')

# Assuming PredictiveBlock and WorldModelGraph are already defined as above.

# --- Step 1: Create the graph with a circular dependency (temporal resolution) ---

# Suppose the environment observation space is a vector (like in CartPole)
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
depth_of_thought = 8
# Each block will predict next observation dimension as a simplistic target
output_dim = obs_dim
# hidden_dim = 32

print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

sensor_in = SensorAdapterIn(1, depth_of_thought, device=device)
sensor_out = SensorAdapterOut(1, depth_of_thought, device=device)
sensor_adapter = SensorAdapter(sensor_in, sensor_out)
block_factory = lambda: AttentionBlock(d_model=depth_of_thought, output_length=8, device=device)
sensor_world = TwoForwardOneBackBlock(block_factory, lr=1e-10, is_outer=False, sensor_adapter=sensor_adapter)

block_factory_two = lambda: AttentionBlock(d_model=depth_of_thought, output_length=16, device=device)
level_two_block_factory = lambda: TwoForwardOneBackBlock(block_factory_two, lr=1e-10, is_outer=True)
world_model = TwoForwardOneBackWithSensor(level_two_block_factory, lr=1e-10, is_outer=True, sensor_adapter=sensor_adapter)

world_model.to(device)

# Policy and Value blocks:
policy_block = PolicyBlock(depth_of_thought * obs_dim * 4, action_dim).to(device)
policy_optimizer = optim.Adam(policy_block.parameters(), lr=1e-8)

value_block = ValueBlock(depth_of_thought * obs_dim * 4).to(device)
value_optimizer = optim.Adam(value_block.parameters(), lr=1e-8)

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
        latent = world_model(input_to_model)  # (batch, 16, 16)
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
            next_latent = world_model.eval_no_backward(next_input_to_model).detach()
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
                if episodes_completed % 100 == 0:
                    print(f"Episode: {episodes_completed}, Reward MA: {ma}")
                episode_rewards[i] = 0.0  # reset for next
                if episodes_completed >= max_episodes:
                    break

        # Optional: print progress
        if step_count % 100 == 0:
            print(f"Step {step_count}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, World Model Loss: {world_model.loss:.4f}")
# def train():
#
#     for ep in range(num_episodes):
#         obs, info = envs.reset(seed=ep)
#         obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(1)  # [1, obs_dim]
#
#         log_probs = []
#         values = []
#         rewards_list: List[torch.Tensor] = []
#         autoreset = torch.zeros(num_envs, dtype=torch.bool, device=device)
#         for t in range(max_steps):
#
#             enriched_from_sensor = sensor_block(obs).permute(0, 2, 1)  # reshape to
#
#             out = model.forward(enriched_from_sensor)
#             test = out.flatten(1)
#             action_logits = policy_block(out.flatten(1).detach())
#             value_pred = value_block(out.flatten(1).detach())  # shape [1, 1]
#
#             # Sample action
#             dist = Categorical(logits=action_logits)
#             action = dist.sample()  # shape [1]
#
#             # Step env
#             next_obs, rewards, terminations, truncations, infos = envs.step(np.array(action.cpu().numpy()))
#             next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(1)
#
#             # Store log_prob, value, reward
#             log_prob = dist.log_prob(action)
#             log_probs.append(log_prob)
#             values.append(value_pred)
#             rewards_list.append(rewards)
#
#             obs = next_obs_t
#
#             autoreset = torch.logical_or(torch.Tensor(terminations), torch.Tensor(truncations))
#
#             if autoreset.all():
#                 break
#
#         # Compute returns and advantages
#         total_reward = sum(rewards).item()
#         ma.add_value(total_reward)
#         returns = []
#         G = 0
#         for r in reversed(rewards):
#             G = r.item() + gamma * G
#             returns.insert(0, G)
#         returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)  # [T, 1]
#
#         values = torch.cat(values, dim=0)  # [T, 1]
#         log_probs = torch.cat(log_probs, dim=0)  # [T]
#
#         # Advantage = returns - values
#         advantages = returns - values.detach()
#
#         # Policy loss: -log_prob * advantage
#         policy_loss = -(log_probs * advantages).mean()
#
#         # Value loss: MSE(returns, values)
#         value_loss = (values - returns).pow(2).mean()
#
#         # Optional: If you still want to encourage good prediction performance,
#         # you could also add a prediction loss using graph.get_predictions at each timestep.
#         # For now, we focus on RL signals only.
#
#         optimizer.zero_grad()
#         loss = policy_loss + 0.5 * value_loss
#         loss.backward()
#         optimizer.step()
#
#         print(f"Episode {ep}: Reward={total_reward}, Reward MA={ma}, Return"
#               f"={returns.sum().item():.2f}, "
#               f"Policy Loss"
#               f"={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}")
#
#     envs.close()

train()

# # Profile the training function
# cProfile.run('train()', 'profile_output')
#
# # Print profiling results
# with open('profile_results.txt', 'w') as f:
#     p = pstats.Stats('profile_output', stream=f)
#     p.sort_stats('cumulative').print_stats(50)
