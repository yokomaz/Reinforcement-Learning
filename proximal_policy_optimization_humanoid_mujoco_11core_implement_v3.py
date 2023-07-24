# Import packages

import random
import time
import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import csv

# Environment wrappering

def make_env(env_name, seed):
  def thunk():
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    #env = gym.wrappers.NormalizeReward(env)
    #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
  return thunk

"""# Layer init"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

"""# Agent"""

class Agent(nn.Module):
  def __init__(self, envs):
    super(Agent, self).__init__()
    self.critic = nn.Sequential(
      layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
      nn.Tanh(),
      layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      layer_init(nn.Linear(64, 1), std=1.0),
    )
    self.actor_mean = nn.Sequential(
        layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
    )
    self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

  def get_value(self, x):
    return self.critic(x)

  def get_action_and_value(self, x, action=None):
    action_mean = self.actor_mean(x)
    action_logstd = self.actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

"""# Test env"""

def test_env(env_name, model, update_idx):
  env = gym.vector.SyncVectorEnv([make_env(gym_env_name, seed) for i in range(num_envs)])
  obs, _ = env.reset()
  obs = torch.tensor(obs, dtype=torch.float32)
  done = False
  score = 0

  while not done:
    action, _, _, _ = model.get_action_and_value(obs)
    obs_, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
    done = terminated or truncated
    score += reward
    obs = torch.tensor(obs_, dtype=torch.float32)

  print(f"episode {update_idx} score is: {score}")
  env.close()
  return score

"""# Save hyperparameters into csv file"""

def save_hyperparameters(env_name,
                         learning_rate,
                         seed,
                         total_timesteps,
                         num_steps_per_env,
                         anneal_lr,
                         gae_lambda,
                         gamma,
                         num_minibatches,
                         update_epoches,
                         clip_coef,
                         entropy_loss_coef,
                         value_loss_coef,
                         max_grad_norm,
                         batch_size,
                         minibatch_size):
  data = [
      ['Env', 'Learning rate', 'seed', 'total_timesteps', 'Num steps per env', 'Anneal learning rate',
       'Gae lambda', 'Gamma', 'Num minibatches', 'Update epoches', 'Clip coef', 'Entropy loss coef',
       'Value loss coef', 'Max gradient norm', 'Batch size', 'Minibatch size'],
      [str(env_name), str(learning_rate), str(seed), str(total_timesteps), str(num_steps_per_env), str(anneal_lr),
       str(gae_lambda), str(gamma), str(num_minibatches), str(update_epoches), str(clip_coef), str(entropy_loss_coef),
       str(value_loss_coef), str(max_grad_norm), str(batch_size), str(minibatch_size)]
  ]

  filename = "Hyperparameters.csv"

  with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in data:
      writer.writerow(row)

"""# Main"""

if __name__ == '__main__':
  # Hyperparameters
  gym_env_name = "Humanoid-v4"
  learning_rate = 3e-4
  seed = 1
  total_timesteps = 2000000
  num_envs = 1
  num_steps_per_env = 2048
  anneal_lr = True
  gae_lambda = 0.95
  gamma = 0.99
  num_minibatches = 32
  update_epoches = 10
  clip_coef = 0.2
  entropy_loss_coef = 0
  value_loss_coef = 0.5
  max_grad_norm = 0.5
  batch_size = int(num_envs * num_steps_per_env)
  minibatch_size = int(batch_size // num_minibatches)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



  # Set seeds
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  envs = gym.vector.SyncVectorEnv(
      [make_env(gym_env_name, seed) for i in range(num_envs)]
      )
  agent = Agent(envs)
  optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

  # storage setup
  observations = torch.zeros((num_steps_per_env, num_envs) + envs.single_observation_space.shape).to(device)
  actions = torch.zeros((num_steps_per_env, num_envs) + envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((num_steps_per_env, num_envs)).to(device)
  rewards = torch.zeros((num_steps_per_env, num_envs)).to(device)
  dones = torch.zeros((num_steps_per_env, num_envs)).to(device)
  values = torch.zeros((num_steps_per_env, num_envs)).to(device)

  global_step = 0
  start_time = time.time()
  observation, _ = envs.reset()
  observation = torch.tensor(observation, dtype=torch.float32).to(device)
  done = torch.zeros(num_envs).to(device)
  num_updates = 100000
  reward_log = []
  print(f"num_update: {num_updates}, batch_size:{batch_size}, minibatch_size:{minibatch_size}")

  for update in range(1, num_updates):
    update_reward = 0

    if anneal_lr:
      frac = 1.0 - (update - 1.0) / num_updates
      lrnow = frac * learning_rate
      optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, num_steps_per_env):
      global_step += 1 * num_envs
      observations[step] = observation
      dones[step] = done

      with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(observation)
        values[step] = value.flatten()
      actions[step] = action
      logprobs[step] = logprob

      observation_, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
      rewards[step] = torch.tensor(reward).to(device).view(-1)
      done = torch.tensor(int(terminated or truncated)).to(device)
      observation = torch.tensor(observation_, dtype=torch.float32).to(device)
      update_reward += reward

    with torch.no_grad():
      next_value = agent.get_value(observation).reshape(1, -1)
      advantages = torch.zeros_like(rewards).to(device)
      last_gae_lambda = 0
      for t in reversed(range(num_steps_per_env)):
        if t == num_steps_per_env - 1:
          next_non_terminal = 1 - done
          nextvalues = next_value
        else:
          next_non_terminal = 1.0 - dones[t+1]
          nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * next_non_terminal - values[t]
        advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
      returns = advantages + values

    batch_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
    batch_logprobs = logprobs.reshape(-1)
    batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    batch_advantages = advantages.reshape(-1)
    batch_returns = returns.reshape(-1)
    batch_values = values.reshape(-1)

    batch_idx = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epoches):
      np.random.shuffle(batch_idx)
      for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        minibatch_idx = batch_idx[start:end]

        _, new_log_prob, entropy, new_value = agent.get_action_and_value(batch_obs[minibatch_idx], batch_actions[minibatch_idx])
        log_ratio = new_log_prob - batch_logprobs[minibatch_idx]
        ratio = log_ratio.exp()

        minibatch_advantages = batch_advantages[minibatch_idx]

        # policy loss
        policy_gradient_loss_1 = -minibatch_advantages * ratio
        policy_gradient_loss_2 = -minibatch_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        policy_gradient_loss = torch.max(policy_gradient_loss_1, policy_gradient_loss_2).mean()

        # value loss
        value_loss = 0.5 * ((new_value - batch_returns[minibatch_idx]) ** 2).mean()

        # entropy loss
        entropy_loss = entropy.mean()

        # total loss
        total_loss = policy_gradient_loss + value_loss_coef * value_loss - entropy_loss_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm(agent.parameters(), max_grad_norm)
        optimizer.step()

    reward_log.append(test_env(gym_env_name, agent, update))

  # Plot reward after training and save the result
  img_name = f'Proximal policy optimization Humanoid {learning_rate}'
  x = np.arange(len(reward_log))
  plt.plot(x, reward_log, color='blue', label='episodic_score')
  plt.xlabel('episodes')
  plt.ylabel('score')
  plt.legend()
  plt.title(img_name)
  plt.show()
  plt.savefig("PPO humanoid", dpi=300)

  # Save model
  model_path = "model.pth"
  torch.save(agent, model_path)

  # Save hyperparameters
  save_hyperparameters(env_name=gym_env_name,
                       learning_rate=learning_rate,
                       seed=seed,
                       total_timesteps=total_timesteps,
                       num_steps_per_env=num_steps_per_env,
                       anneal_lr=anneal_lr,
                       gae_lambda=gae_lambda,
                       gamma=gamma,
                       num_minibatches=num_minibatches,
                       update_epoches=update_epoches,
                       clip_coef=clip_coef,
                       entropy_loss_coef=entropy_loss_coef,
                       value_loss_coef=value_loss_coef,
                       max_grad_norm=max_grad_norm,
                       batch_size=batch_size,
                       minibatch_size=minibatch_size)

  # Load model and test
  model = torch.load(model_path)
  test_env(gym_env_name, model, update_idx='Test')

  # Close env
  envs.close()
