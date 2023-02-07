import torch
import torch.nn as nn
import torch.nn.functional as F
from pusoy.losses import identity, state_value, q_value, state_value_advantage, q_value_advantage

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

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

class D2RLActor(torch.nn.Module):
    """
    Model that takes input and returns output logits vector.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)
    
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_2(x))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_3(x))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_4(x))

        return self.out_layer(x)

class D2RLCritic(torch.nn.Module):
    """
    Model that takes input and returns output logits vector.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

        with torch.no_grad():
            self.out_layer.weight /= 100
    
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_2(x))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_3(x))
        x = torch.cat([x, state], dim=-1)
        x = F.relu(self.layer_4(x))

        return self.out_layer(x)

class Base(nn.Module):
    """
    Model that uses D2RL architecture to produce an action.
    Winning actions are given positive reward.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__()
        self.actor = D2RLActor(hidden_dim, input_dim, output_dim)
        self.critic = None
        self.adv_func = identity


class D2RLAC(Base):
    """
    Model that uses D2RL architecture to produce an action.
    Critic evaluates how good a state is.
    Actions that lead to good states are given rewards.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.critic = D2RLCritic(hidden_dim, input_dim, 1)
        self.adv_func = state_value

class D2RLA2C(D2RLAC):
    """
    Uses advantage in its reward function instead of state value.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.adv_func = state_value_advantage

class D2RLAQC(Base):
    """
    Model that uses D2RL architecture to produce an action.
    Critic evaluates how good an action is directly.
    Actions that produce winning conditions are given rewards.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.critic = D2RLCritic(hidden_dim, input_dim + output_dim, 1)
        self.adv_func = q_value


class D2RLA2QC(D2RLAC):
    """
    Uses advantage in its reward function instead of state value.
    """
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__(hidden_dim, input_dim, output_dim)
        self.adv_func = q_value_advantage