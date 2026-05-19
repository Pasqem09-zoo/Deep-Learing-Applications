"""
Utility functions used across the project.

Includes:
- action selection from policy
- return computation for policy gradient methods
- seed setting for reproducibility
"""

import torch
import numpy as np
import random
from torch.distributions import Categorical


def select_action(policy, state):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy(state)
    dist = Categorical(action_probs)
    action = dist.sample()

    return action.item(), dist.log_prob(action)



def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return returns



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)