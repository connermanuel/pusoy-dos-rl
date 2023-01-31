from pusoy.train import kl_loss
from pusoy.action import PlayCards
from pusoy.utils import RoundType, Hands
import pytest
import torch
from pusoy.decision_function import Neural

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
    curr_model, prev_model = model, model
    input = torch.rand((330,))
    output = curr_model(input)
    loss = kl_loss(curr_model, prev_model, input, 
                    action = PlayCards(torch.ones(52,), RoundType.SINGLES, Hands.NONE), 
                    adv=1, device='cpu')
    opt = torch.optim.SGD(curr_model.parameters(), lr=0.1)

    opt.zero_grad()
    loss.mean().backward()
    opt.step()

    new_output = curr_model(input)

    assert torch.all(torch.ge(new_output, output))

