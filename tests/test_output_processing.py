from pusoy.decision_function import Neural
import torch
import pytest

# ------------- FIXTURES -----------------
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(326, 62)
    
    def forward(self, x):
        return self.layer_1(x)

class RiggedModel(DummyModel):
    def __init__(self, output):
        super().__init__()
        self.output = output
    
    def forward(self, x):
        return self.output
    
@pytest.fixture
def nn():
    model = DummyModel()
    return Neural(model)

# ----------------------------------------

def test_find_pairs_base(nn):
    val = torch.randint(13, (1,))[0]
    idxs = torch.randint(4, (2,))
    idxs = idxs + (val*4)
    output = torch.zeros(62)
    output[idxs] = 1
    card_list = torch.ones(52)
    prev_play = torch.zeros(52)
    hand_type = torch.zeros(5)
    is_pending = 1

    best_pair = nn.find_best_pair(output, card_list, prev_play, hand_type, is_pending)
    assert all(best_pair.cards == output[:52])

def test_find_pairs_with_prev_play(nn):
    vals = torch.sort(torch.randint(13, (2,)))[0]
    idxs = torch.randint(4, (2,))
    curr_idxs = idxs + (vals[0]*4)
    prev_idxs = idxs + vals[1]*4
    output = torch.zeros(62)
    output[curr_idxs] = 1
    card_list = torch.ones(52)
    prev_play = torch.zeros(52)
    prev_play[prev_idxs] = 1
    hand_type = torch.zeros(5)
    is_pending = 0

    best_pair = nn.find_best_pair(output, card_list, prev_play, hand_type, is_pending)
    if best_pair:
        print(best_pair.cards)
    assert best_pair is None

def test_play_pairs_with_prev_play():
    round_type = torch.zeros(5)
    round_type[2] = 1
    hand_type = torch.zeros(5)
    curr_player = 0
    prev_player = 3
    card_list = torch.zeros(52)
    card_list[4:6] = 1
    prev_play = torch.zeros(52)
    prev_play[0:2] = 1
    played_cards = [torch.zeros(52)] * 4

    output = torch.cat([card_list, round_type[1:], hand_type, torch.zeros(1)])

    rigged_model = Neural(RiggedModel(output))

    resulting_action = rigged_model.play(
        curr_player, card_list, round_type, hand_type, prev_play, prev_player, played_cards
    )

    assert all(resulting_action.cards == card_list[:52])

# def test_play_pairs_with_pending_round_type():
