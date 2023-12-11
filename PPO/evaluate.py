import gymnasium as gym
import random
import numpy as np
import torch 
from experts.PG import PG

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate_cartpole():
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

    observation, info = env.reset(seed=18095048)
    count = 0

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    policy = PG(state_shape, n_actions)
    policy.load_state_dict(torch.load('policy_model.pth'))

    for _ in range(2000):
        states = torch.tensor(observation, dtype=torch.float32)
        action = policy(states).detach().numpy()
        print(action)
        idx = np.argmax(action)

        observation, reward, terminated, truncated, info = env.step(idx)
        
        if terminated or truncated:
            print(">>>>>>>>>>>>>>>>>>")
            print(_, terminated, truncated)
            observation, info = env.reset()
            break
    env.close()

def evaluate_acrobot():
    env = gym.make('Acrobot-v1', render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

    observation, info = env.reset(seed=18095048)
    count = 0

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    policy = PG(state_shape, n_actions)
    policy.load_state_dict(torch.load('acrobot_policy_model.pth'))

    for _ in range(2000):
        states = torch.tensor(observation, dtype=torch.float32)
        action = policy(states).detach().numpy()
        print(action)
        idx = np.argmax(action)

        observation, reward, terminated, truncated, info = env.step(idx)
        
        if terminated or truncated:
            print(">>>>>>>>>>>>>>>>>>")
            print(_, terminated, truncated)
            observation, info = env.reset()
            break
    env.close()

def visulize_ppo():
    pass

evaluate_acrobot()