"""
Neural network architectures used in the reinforcement learning experiments.

This module defines:
- PolicyNet: a policy network that maps a state to action probabilities. 
  It is used as the policy in both REINFORCE and A2C.
- ValueNet: a value network that maps a state to a scalar value
  estimate V(s). It is used as a learned baseline in REINFORCE and as the
  critic in A2C.
"""


import torch.nn as nn
import torch.nn.functional as F



#Simple feedforward policy network to approximate the policy. Given a state, it outputs a probability distribution over the available discrete actions.
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim) # maps the input state to a hidden representation
        self.fc2 = nn.Linear(hidden_dim, action_dim) # produces one score for each possible action

    def forward(self, x):
        x = F.relu(self.fc1(x))

        # Softmax converts action scores into action probabilities
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs



# Baseline value network to approximate the state value function. Given a state, it outputs a scalar estimate of the state value V(s).
# With baseline we can compute the advantage and reduce the variance of the policy gradient estimate in REINFORCE with baseline.
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
      super().__init__()

      self.fc1 = nn.Linear(state_dim, hidden_dim) # maps the input state to a hidden representation
      self.fc2 = nn.Linear(hidden_dim, 1) # produces a single scalar value representing the estimated value of the input state

    def forward(self, x):
      x = F.relu(self.fc1(x))
      value = self.fc2(x)
      return value