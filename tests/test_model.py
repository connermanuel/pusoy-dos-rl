import torch


def test_model_basic(base_model_a2c):
    input = torch.rand(330)
    _ = base_model_a2c.forward(input)
    pass
