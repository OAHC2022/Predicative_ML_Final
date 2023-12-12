import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from PG import PG
from cost import CostNN
from utils import to_one_hot, get_cumulative_rewards
import json 

from torch.optim.lr_scheduler import StepLR

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ENV SETUP
env_name = 'Acrobot-v1'
env = gym.make(env_name).unwrapped

n_actions = env.action_space.n
state_shape = env.observation_space.shape
state = env.reset(seed=seed)

# LOADING EXPERT/DEMO SAMPLES
with open("../data/gen_data.json", "r") as f: 
    demo_trajs = json.load(f)

# INITILIZING POLICY AND REWARD FUNCTION
policy = PG(state_shape, n_actions)
# policy.load_state_dict(torch.load('bc_pretrain_policy.pth'))

cost_f = CostNN(state_shape[0] + 1)
policy_optimizer = torch.optim.AdamW(policy.parameters(), 8e-4)
cost_optimizer = torch.optim.AdamW(cost_f.parameters(), 8e-4, weight_decay=1e-4)

mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 100
sample_trajs = []

D_demo, D_samp = np.array([]), np.array([])

# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo = False):
    step_list = step_list.tolist()
    for traj in traj_list:
        # traj[0] states list, traj[1] action list
        states = np.array(traj[0])
        if is_Demo:
            probs = np.ones((states.shape[0], 1))
        else:
            probs = np.array(traj[1]).reshape(-1, 1) 
        # probs[probs == 0] = 1 # convert 0 to 1
        actions = np.array(traj[2]).reshape(-1, 1) 
        x = np.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return np.array(step_list)

D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True) # states, prob, actions
# print(D_demo.shape)
# exit()
return_list, sum_of_cost_list = [], []
best_return = -1000
for i in range(200):
    print("=== Training Iteration %d ==="%(i))
    # use default policy to generate more examples and add to the D-sample
    trajs = [policy.generate_session(env,2000) for _ in range(EPISODES_TO_PLAY)]
    sample_trajs = trajs + sample_trajs
    D_samp = preprocess_traj(trajs, D_samp)

    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
    loss_rew = []
    for _ in range(REWARD_FUNCTION_UPDATE):
        selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
        selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

        D_s_samp = D_samp[selected_samp]
        D_s_demo = D_demo[selected_demo]

        #D̂ samp ← D̂ demo ∪ D̂ samp
        D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0)
        
        states, probs, actions = D_s_samp[:,:-2], D_s_samp[:,-2], D_s_samp[:,-1]

        states_expert, actions_expert = D_s_demo[:,:-2], D_s_demo[:,-1]
        # Reducing from float64 to float32 for making computaton faster
        states = torch.tensor(states, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        states_expert = torch.tensor(states_expert, dtype=torch.float32)
        actions_expert = torch.tensor(actions_expert, dtype=torch.float32)

        costs_samp = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1))
        costs_demo = cost_f(torch.cat((states_expert, actions_expert.reshape(-1, 1)), dim=-1))

        # print(costs_samp.shape)
        # exit()
        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        # loss_IOC = torch.mean(costs_demo) + \
        #         torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
        # print(probs) # this is the problem!!!
        # exit()
        # Log-Sum-Exp Trick
        min_value = 1e-7  # You can adjust this value
        clamped_probs = probs + min_value
        max_cost = torch.max(-costs_samp)
        exp_terms = torch.exp(-costs_samp - max_cost) / clamped_probs
        log_mean_exp = torch.log(torch.mean(exp_terms)) + max_cost

        loss_IOC = torch.mean(costs_demo) + log_mean_exp

        # UPDATING THE COST FUNCTION
        cost_optimizer.zero_grad()
        loss_IOC.backward()
        cost_optimizer.step()

        loss_rew.append(loss_IOC.detach())

    for traj in trajs:
        states, probs, actions, rewards = traj
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
            
        costs = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()

        cumulative_returns = np.array(get_cumulative_rewards(-costs, 0.99))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

        logits = policy(states)

        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)
        log_probs_for_actions = torch.sum(
            log_probs * to_one_hot(actions, env.action_space.n), dim=1)
        entropy = -torch.mean(torch.sum(probs*log_probs, dim = -1) )
        loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*1e-2) 

        # UPDATING THE POLICY NETWORK
        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

    returns = sum(rewards)
    sum_of_cost = np.sum(costs)

    if returns > best_return:
        best_return = returns
        torch.save(policy.state_dict(), 'acrobot_policy_model.pth')
    
    return_list.append(returns)
    sum_of_cost_list.append(sum_of_cost)

    mean_rewards.append(np.mean(return_list))
    mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))

    # PLOTTING PERFORMANCE
    if i % 10 == 0:
        # clear_output(True)
        print(f"mean reward:{best_return} loss: {loss_IOC}")

        plt.figure(figsize=[16, 12])
        plt.title(f"Mean reward per {EPISODES_TO_PLAY} games")
        plt.plot(return_list)
        plt.grid()

        # plt.subplot(2, 2, 2)
        # plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
        # plt.plot(mean_costs)
        # plt.grid()

        # plt.subplot(2, 2, 3)
        # plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
        # plt.plot(mean_loss_rew)
        # plt.grid()

        # plt.show()
        plt.savefig('plots/GCL_learning_curve.png')
        plt.close()

with open("rewards.json", "w") as f:
    json.dump(return_list, f)
# print(policy.state_dict())
# torch.save(policy.state_dict(), 'acrobot_policy_model_1000.pth')