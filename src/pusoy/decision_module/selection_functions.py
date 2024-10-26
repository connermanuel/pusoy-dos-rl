from typing import Callable
import torch


SelectionFunction = Callable[[torch.Tensor, int], torch.Tensor]

def selection_function_train(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Given a tensor of probabilities, return an ordered tensor of indices.
    During training, the selection function is a multinomial distribution.
    """
    return torch.multinomial(probs, num_samples, replacement=True)

def selection_function_eval(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Given a tensor of probabilities, return an ordered tensor of indices.
    During evaluation, the selection function is a maximum function.
    """
    return torch.topk(input=probs, k=num_samples)[1]