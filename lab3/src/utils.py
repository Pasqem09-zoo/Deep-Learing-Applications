"""
Utility functions used across the reinforcement learning project.

This file contains small helper functions shared by REINFORCE and A2C:
- stochastic action selection during training
- greedy action selection during evaluation
- discounted return computation
- seed setting for reproducibility
"""

import torch
import numpy as np
import random
from torch.distributions import Categorical


def select_action(policy, state, return_dist=False):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy(state)
    dist = Categorical(action_probs)
    action = dist.sample()

    if return_dist:
        return action.item(), dist.log_prob(action), dist
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



def select_greedy_action(policy, state):
    state = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action_probs = policy(state)
        action = torch.argmax(action_probs)

    return action.item()