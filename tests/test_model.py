import torch
from pusoy.utils import indexes_to_one_hot, RoundType
from pusoy.models import get_probs_from_logits

def test_model_basic(base_model_a2c):
    """Tests that the base_model_a2c can be initialized and forward pass can be called."""
    input = torch.rand(330)
    _ = base_model_a2c.forward(input)
    pass

def test_get_probs_from_logits():
    """Tests that the get_probs_from_logits function returns the correct shapes and values."""
    logits = torch.rand(330)
    card_list = indexes_to_one_hot(size=52, idxs = [0])
    round_type = RoundType.SINGLES
    
    card_probs, round_probs, _= get_probs_from_logits(logits, card_list, round_type)
    
    assert card_probs.shape == (52,)
    assert round_probs.shape == (5,)
    assert card_probs.sum().item() == 1.0
    assert round_probs.sum().item() == 1.0
    assert card_probs[0].item() == 1.0
    assert round_probs[0:2].sum() == 1.0
