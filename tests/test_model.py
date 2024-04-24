import torch

from pusoy.models import DenseA2C


def test_model_basic():
    input = torch.rand(330)
    model = DenseA2C()
    _ = model.forward(input)
    pass
