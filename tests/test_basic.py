import torch
from pusoy.models import DenseA2C
from pusoy.losses import gae

def test_gae():
    state_values = torch.rand(15)
    rewards = torch.zeros(15)
    rewards[-1] = 1

    gae(state_values, rewards)