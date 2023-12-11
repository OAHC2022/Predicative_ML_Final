import torch 
import torch.nn as nn
import numpy as np
import gymnasium as gym
from network import FeedForwardNN
import json 

env = gym.make('Acrobot-v1', render_mode="rgb_array")


obs_dim = env.observation_space.shape[0]
act_dim = 3
policy = FeedForwardNN(obs_dim, act_dim)
policy.load_state_dict(torch.load('ppo_actor.pth'))


# env = gym.wrappers.RecordEpisodeStatistics(env)
# env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

data = []
observation, info = env.reset()
count = 0
num_expert = 100
states, traj_probs, actions, rewards = [], [], [], []
while True:
    action = policy(observation).detach().numpy()
    idx = np.argmax(action)
    new_observation, reward, terminated, truncated, info = env.step(idx)

    tmp1 = observation.tolist()
    tmp2 = int(idx)
    tmp3 = int(reward)
    states.append(tmp1)
    actions.append(tmp2)
    rewards.append(tmp3)

    observation = new_observation
    if terminated or truncated:
        observation, info = env.reset()
        data.append([states, rewards, actions])
        
        count += 1
    if count >= num_expert:
        break
env.close()
# s, r, a = data[0]
# print(len(s), len(r), len(a))
# exit()
# print(data.shape)
with open('data/gen_data.json', 'w') as f:
    json.dump(data, f)
# np.save(f'data/gen_data.npy', data)
# print(policy)
