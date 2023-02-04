import torch
import torch.nn as nn
import torch.nn.functional as F
from pusoy.losses import identity, state_value, q_value, state_value_advantage, q_value_advantage

class DumbModel(torch.nn.Module):
    """
    Baseline dumb model. For testing only.
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(330, 62)
        self.adv_func = identity
    
    def forward(self, x):
        return self.layer_1(x) # Returns logits

class D2RL(torch.nn.Module):
    """
    Model that uses D2RL architecture to produce an action.
    Winning actions are given positive reward.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

        with torch.no_grad():
            self.out_layer.weight /= 100
        
        self.adv_func = identity
    
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_2(x))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_3(x))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_4(x))

        return self.out_layer(x)

class D2RLAC(D2RL):
    """
    Model that uses D2RL architecture to produce an action.
    Critic evaluates how good a state is.
    Actions that lead to good states are given rewards.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.adv_func = state_value

class D2RLA2C(D2RLAC):
    """
    Uses advantage in its reward function instead of state value.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.adv_func = state_value_advantage

class D2RLAQC(D2RL):
    """
    Model that uses D2RL architecture to produce an action.
    Critic evaluates how good an action is directly.
    Actions that produce winning conditions are given rewards.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.critic = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.adv_func = q_value


class D2RLA2QC(D2RLAC):
    """
    Uses advantage in its reward function instead of state value.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.adv_func = q_value_advantage