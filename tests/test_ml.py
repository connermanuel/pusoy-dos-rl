from pusoy.losses import batch_loss, logits_to_log_probs
from pusoy.action import PlayCards
from pusoy.utils import RoundType, Hands
import pytest
import torch
import copy

# ------------- FIXTURES -----------------
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(330, 62)
    
    def forward(self, x):
        return self.layer_1(x)

class RiggedModel(DummyModel):
    def __init__(self, output):
        super().__init__()
        self.output = output
    
    def forward(self, x):
        return self.output
    
@pytest.fixture
def model():
    return DummyModel()
# ------------- FIXTURES -----------------

def test_loss(model):
    for _ in range(2):
        curr_model = model
        prev_model = copy.deepcopy(model)
        input = torch.ones((330,))
        opt = torch.optim.SGD(curr_model.parameters(), lr=0.01)
        base = prev_model(input).clone()

        # Create a dummy action that plays a single random card
        idx = torch.randint(52, (1,)).item()
        out_mask = torch.zeros(62, dtype=torch.int)
        out_mask[idx] = 1
        out_mask[53] = 1
        out_mask[57] = 1
        out_mask = out_mask.bool()
        rev_mask = ~out_mask
        cards = torch.zeros(52)
        cards[idx] = 1
        action = PlayCards(cards, RoundType.SINGLES, Hands.STRAIGHT)

        for i in range(10):
            loss = batch_loss(curr_model, prev_model, inputs=[input], 
                            actions=[action], adv=1, device='cpu')
            opt.zero_grad()
            loss.sum().backward()
            opt.step()

        new_output = curr_model(input)
        new_output = logits_to_log_probs(new_output[None, ...]).flatten()
        base = logits_to_log_probs(base[None, ...]).flatten()

        print(f"index: {idx}")
        print(f"Difference: {new_output - base}")
        assert torch.all(torch.gt(new_output[out_mask], base[out_mask]))
        assert torch.all(torch.lt(new_output[rev_mask], base[rev_mask]))

