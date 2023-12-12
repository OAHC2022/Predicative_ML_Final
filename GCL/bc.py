from PG import PG
import gymnasium as gym
import json 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ENV SETUP
env_name = 'Acrobot-v1'
env = gym.make(env_name).unwrapped

n_actions = env.action_space.n
state_shape = env.observation_space.shape
policy = PG(state_shape, n_actions)

# LOADING EXPERT/DEMO SAMPLES
with open("../data/gen_data.json", "r") as f: 
    demo_trajs = json.load(f)


class SimpleDataset(Dataset):
    """A simple dataset of random tensors and labels."""

    def __init__(self, step_list):
        self.data = step_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0].astype(np.float32), self.data[idx][1]
    
# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list):
    step_list = []
    for traj in traj_list:
        # traj[0] states list, traj[1] action list
        
        for i in range(len(traj[0])):
            states = np.array(traj[0][i])
            actions = np.array(traj[2][i])
            step_list.append([states, actions])
    return step_list


step_list = preprocess_traj(demo_trajs)

dataset = SimpleDataset(step_list)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
policy_optimizer = torch.optim.AdamW(policy.parameters(), 1e-3)


# Assuming 'dataloader' is defined (as shown in the previous code)
for epoch in range(2):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        # Zero the parameter gradients
        policy_optimizer.zero_grad()

        # Forward pass
        outputs = policy(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        policy_optimizer.step()

        running_loss += loss.item()

    # Print statistics
    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}")
    running_loss = 0.0

torch.save(policy.state_dict(), 'bc_pretrain_policy.pth')
print('Finished Training')