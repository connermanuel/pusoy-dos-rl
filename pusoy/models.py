import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pusoy.constants import OUTPUT_SIZES

class PusoyModel(nn.Module, ABC):
    """Defines the model interface for Pusoy models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def v(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

class DenseA2C(nn.Module):
    """A dense model. Generates a common feature vector that is used as the baseline for both actor and critic models."""

    def __init__(self, input_size=330, hidden_size=256, output_size=62):
        super().__init__()
        self.preprocess = nn.Linear(in_features=input_size, out_features=hidden_size)

        self.actor_1 = nn.Linear(hidden_size, hidden_size)
        self.actor_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.actor_out = nn.Linear(hidden_size, output_size)
        
        self.critic_1 = nn.Linear(hidden_size, hidden_size)
        self.critic_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor]:
        """Generates an action vector from an input state.
        
        Input:
            state - (batch_size, state_dim) A tensor representing state spaces
        
        Output:
            A tuple of vectors of logits over the action space.
        """
        x = self.preprocess(state)
        
        actor_x = F.relu(self.actor_1(x))
        actor_x = torch.cat([actor_x, x], dim=-1)
        actor_x = F.relu(self.actor_2(actor_x))
        actor_out = self.actor_out(actor_x)
        actor_out = torch.split(actor_out, OUTPUT_SIZES, dim=-1)

        return actor_out
    
    def forward(self, state) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        """Generates an action vector and a state evalution from .
        
        Input:
            state - (batch_size, state_dim) A tensor representing state spaces
        
        Output:
            A tuple containing a tuple of vectors representing action logits for the different spaces,
            and an evaluation over the states (batch_size,).
        """
        x = self.preprocess(state)
        
        actor_x = F.relu(self.actor_1(x))
        actor_x = torch.cat([actor_x, x], dim=-1)
        actor_x = F.relu(self.actor_2(actor_x))
        actor_out = self.actor_out(actor_x)
        actor_out = torch.split(actor_out, OUTPUT_SIZES, dim=-1)

        critic_x = F.relu(self.critic_1(x))
        critic_x = torch.cat([critic_x, x], dim=-1)
        critic_x = F.relu(self.critic_2(critic_x))
        critic_out = self.critic_out(critic_x)
        critic_out = critic_out.flatten()

        return actor_out, critic_out
