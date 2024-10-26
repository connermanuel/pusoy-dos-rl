import pytest
import torch

from pusoy.decision_module import Neural, TrainingNeural
from pusoy.models import DenseA2C


@pytest.fixture
def base_model_a2c() -> DenseA2C:
    """Basic nondescript version of a model."""
    return DenseA2C()


@pytest.fixture
def base_training_decision_function(base_model_a2c) -> TrainingNeural:
    """Training df using the base model."""
    return TrainingNeural(base_model_a2c)


@pytest.fixture
def random_hand() -> torch.Tensor:
    """A random hand of 13 cards from a possible 52."""
    idxs = torch.randperm(52)[:13]
    cards = torch.zeros(52)
    cards[idxs] = 1
    return cards

@pytest.fixture
def neural_decision_function(base_model_a2c) -> Neural:
    return Neural(base_model_a2c)