"""
This module defines neural network architectures used in the project.

Currently contains:
- PolicyNet: a simple feedforward network that outputs action probabilities
  given a state, used for policy gradient methods like REINFORCE.
- You can add more architectures here as needed, such as value networks for
  actor-critic methods, or more complex policy networks for larger environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# A simple, but generic, policy network with one hidden layer.
### ci serve un approssimatore della policy quindi usiamo una rete neurale! 
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128): ### prende l'ambiente per capire la dimensione dello spazio di osservazione e di azione
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim) ###in input qualsiasi tipo di dato xk pernde la sua shape, quindi ha 4 valori: un tensore di 4 dimensioni, e un hidden layer con 128 neuroni
        self.fc2 = nn.Linear(hidden_dim, action_dim) ### produce esattamente il numero di output necessari che in questo caso sono due azioni: spostare a sn o a dx

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs