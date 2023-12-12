import gymnasium as gym
import random
import numpy as np
import torch 
from GCL.PG import PG
from PPO.network import FeedForwardNN

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def visualize_acrobot():
    env = gym.make('Acrobot-v1', render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

    observation, info = env.reset(seed=18095048)
    count = 0

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    policy = PG(state_shape, n_actions)
    # policy.load_state_dict(torch.load('GCL/acrobot_policy_model.pth'))
    # policy.load_state_dict(torch.load('GCL/acrobot_policy_model_bc.pth'))
    policy.load_state_dict(torch.load('GCL/bc_pretrain_policy.pth'))

    for _ in range(2000):
        states = torch.tensor(observation, dtype=torch.float32)
        action = policy(states).detach().numpy()
        print(action)
        idx = np.argmax(action)

        observation, reward, terminated, truncated, info = env.step(idx)
        
        if terminated or truncated:
            print(">>>>>>>>>>>>>>>>>>")
            print(count)
            print(_, terminated, truncated)
            observation, info = env.reset()
            count += 1
        if count >= 1:
            break
    env.close()

def evaluate_acrobot_gcl():
    env = gym.make('Acrobot-v1', render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

    observation, info = env.reset(seed=18095048)
    count = 0
    steps = 0

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    policy = PG(state_shape, n_actions)
    policy.load_state_dict(torch.load('GCL/acrobot_policy_model_bc.pth'))
    # policy.load_state_dict(torch.load('GCL/bc_pretrain_policy.pth'))
    # policy.load_state_dict(torch.load('GCL/acrobot_policy_model.pth'))
    total_steps = []
    while True:
        states = torch.tensor(observation, dtype=torch.float32)
        action = policy(states).detach().numpy()
        idx = np.argmax(action)

        observation, reward, terminated, truncated, info = env.step(idx)
        steps += 1
        
        if terminated or truncated:
            total_steps.append(steps)
            steps = 0
            count += 1
            observation, info = env.reset()
        if count >= 500:
            break
    env.close()
    return total_steps


def evaluate_acrobot_ppo():
    env = gym.make('Acrobot-v1', render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

    observation, info = env.reset(seed=18095048)

    obs_dim = env.observation_space.shape[0]
    act_dim = 3
    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load('PPO/ppo_actor.pth'))

    data = []
    total_steps = []
    observation, info = env.reset()
    count = 0
    steps = 0
    num_expert = 500
    while True:
        action = policy(observation).detach().numpy()
        idx = np.argmax(action)
        new_observation, reward, terminated, truncated, info = env.step(idx)
        steps += 1

        observation = new_observation
        if terminated or truncated:
            total_steps.append(steps)
            steps = 0
            count += 1
            observation, info = env.reset()
            break
            
        if count >= num_expert:
            break
    env.close()
    return total_steps
# visualize_acrobot()
# total_steps = evaluate_acrobot_gcl()
total_steps = evaluate_acrobot_ppo()
# total_steps = -np.array(total_steps)
# print(np.mean(total_steps), np.std(total_steps), np.max(total_steps), np.min(total_steps))
