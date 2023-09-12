import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import unpack_sequence


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class A2CLSTM(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.actor_1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.actor_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, output_dim)
        self.critic_1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.critic_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, input, states=None, compute_critic=True, packed=False):
        x, states = self.lstm(input, states)
        if packed:
            x = torch.cat(unpack_sequence(x), dim=0)
            input = torch.cat(unpack_sequence(input), dim=0)
        x = torch.cat([x, input], dim=-1)

        actor_x = F.relu(self.actor_1(x))
        actor_x = torch.cat([actor_x, input], dim=-1)
        actor_x = F.relu(self.actor_2(x))
        actor_out = self.actor_out(actor_x)

        critic_out = None
        if compute_critic:
            critic_x = F.relu(self.critic_1(x))
            critic_x = torch.cat([critic_x, input], dim=-1)
            critic_x = F.relu(self.critic_2(x))
            critic_out = self.critic_out(critic_x)

        return actor_out, critic_out, states


    
