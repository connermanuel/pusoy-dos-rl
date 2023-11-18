import pytest
import torch

from pusoy.models import FullLSTMModel
from pusoy.utils import Hands, RoundType


@pytest.fixture
def model():
    return FullLSTMModel()


@pytest.fixture
def sample_input():
    card_list = torch.zeros(52)
    prev_play = torch.zeros(52)
    card_idxs = torch.randperm(52)
    for idx in card_idxs[:13]:
        card_list[idx] = 1
    prev_play[card_idxs[13]] = 1

    return {
        "player_no": 0,
        "card_list": card_list,
        "round_type": RoundType.SINGLES,
        "hand_type": Hands.NONE,
        "prev_play": prev_play,
        "prev_player": 3,
        "played_cards": [torch.zeros(52)] * 3 + [prev_play],
        "hx": None,
    }
