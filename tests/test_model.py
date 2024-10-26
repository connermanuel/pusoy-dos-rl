import torch
from pusoy.action import PlayCards
from pusoy.utils import Hands, indexes_to_one_hot, RoundType
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

def test_play_three_of_clubs(neural_decision_function):
    # Create an input tensor where the card list only contains the three of clubs
    card_list = indexes_to_one_hot(52, [0])
    
    # Create other necessary inputs
    round_type = RoundType.NONE
    hand_type = Hands.NONE
    prev_play = torch.zeros(52)
    is_first_move = True
    
    # Run the neural decision function
    action = neural_decision_function.play(
        player_no=0,
        card_list=card_list,
        round_type=round_type,
        hand_type=hand_type,
        prev_play=prev_play,
        prev_player=0,
        played_cards=[torch.zeros(52)] * 4,
        is_first_move=is_first_move
    )
    
    # Assert that the returned action is to play the three of clubs
    assert isinstance(action, PlayCards)
    assert torch.all(action.cards == card_list)
