import torch

from pusoy.losses import ppo_loss
from pusoy.models import DenseA2C


def test_loss_basic():
    input = torch.rand((1, 330))
    model = DenseA2C()

    loss = ppo_loss()
