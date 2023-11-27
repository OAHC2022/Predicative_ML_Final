import torch 
import torch.nn as nn
import numpy as np
import gymnasium as gym
from network import FeedForwardNN
env = gym.make('Acrobot-v1', render_mode="rgb_array")


obs_dim = env.observation_space.shape[0]
act_dim = 3
policy = FeedForwardNN(obs_dim, act_dim)
policy.load_state_dict(torch.load('ppo_actor.pth'))


env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)
observation, info = env.reset()
for _ in range(1000):
    action = policy(observation).detach().numpy()
    idx = np.argmax(action)
    print(action, idx)

    observation, reward, terminated, truncated, info = env.step(idx)

    if terminated or truncated:
        observation, info = env.reset()
    print(observation, reward)
env.close()

print(policy)
