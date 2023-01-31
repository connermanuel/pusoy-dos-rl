import torch
import torch.nn as nn
import torch.nn.functional as F

class DumbModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(330, 62)
    
    def forward(self, x):
        return self.layer_1(x) # Returns logits

class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer_1 = nn.Linear(330, 128)
        layer_2 = nn.Linear(128, 128)
        layer_3 = nn.Linear(128, 128)
        layer_4 = nn.Linear(128, 62)
        self.layer = nn.Sequential(
            layer_1,
            nn.Tanh(),
            layer_2,
            nn.Tanh(),
            layer_3,
            nn.Tanh(),
            layer_4
        )
        torch.nn.init.xavier_uniform_(layer_1.weight)
        torch.nn.init.xavier_uniform_(layer_2.weight)
        torch.nn.init.xavier_uniform_(layer_3.weight)
        torch.nn.init.xavier_uniform_(layer_4.weight)

        with torch.no_grad():
            layer_4.weight /= 100
    
    def forward(self, x):
        return self.layer(x)

class D2RLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(330, 128)
        self.layer_2 = nn.Linear(330+128, 128)
        self.layer_3 = nn.Linear(330+128, 128)
        self.layer_4 = nn.Linear(330+128, 128)
        self.out_layer = nn.Linear(128, 62)

        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

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

class D2RLModelWithCritic(D2RLModel):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(330, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.MSELoss = nn.MSELoss()

class D2RLModelWithQValueCritic(D2RLModel):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(330+62, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.MSELoss = nn.MSELoss()
