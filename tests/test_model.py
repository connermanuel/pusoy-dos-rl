import torch
import torch.nn.functional as F
import pytest
from pusoy.utils import RoundType, Hands

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_basic(model, sample_input, device):
    """Checks that the model can handle an input."""

    model.to(device)
    card_list = sample_input["card_list"]

    out, hx = model.get_hidden_state_from_input(**sample_input)
    card_list_embeddings = model.get_card_list_embeddings(card_list)
    action_probabilities = model.get_action_probabilities(out)
    assert torch.isclose(torch.sum(action_probabilities), torch.tensor(1, dtype=torch.float))
    assert len(action_probabilities) == 9
    for round_type_idx in [1, 2, 3]:
        card_probs = model.get_card_probabilities(out, card_list_embeddings, round_type_idx)
        assert torch.isclose(torch.sum(card_probs), torch.tensor(1, dtype=torch.float))
        assert len(card_probs) == 13

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_backprop_action(model, sample_input, device):
    """Checks that all elements of the model backpropagate successfully with losses in action probs."""

    model.to(device)
    card_list = sample_input["card_list"]
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    out, hx = model.get_hidden_state_from_input(**sample_input)
    out_2, hx = model.get_hidden_state_from_input(**sample_input)
    assert torch.eq(out, out_2).all()

    action_probabilities = model.get_action_probabilities(out)
    rand_action_probabilities = F.softmax(torch.rand(9), dim=0).to(device)
    
    F.mse_loss(action_probabilities, rand_action_probabilities).backward()
    opt.step()

    out_3, hx = model.get_hidden_state_from_input(**sample_input)
    assert not torch.eq(out, out_3).all()

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_backprop_card_probs(model, sample_input, device):
    """Checks that all elements of the model backpropagate successfully with losses in card probs."""
    
    model.to(device)
    card_list = sample_input["card_list"]
    opt = torch.optim.SGD(model.parameters(), lr=1)

    out, hx = model.get_hidden_state_from_input(**sample_input)
    card_list_embeddings = model.get_card_list_embeddings(card_list)
    card_probs = model.get_card_probabilities(out, card_list_embeddings, 1)
    rand_card_probs = F.softmax(torch.rand(13), dim=0).to(device)

    F.mse_loss(card_probs, rand_card_probs).backward()
    opt.step()
    
    out_2, hx = model.get_hidden_state_from_input(**sample_input)
    card_list_embeddings_2 = model.get_card_list_embeddings(card_list)
    assert not torch.allclose(card_list_embeddings, card_list_embeddings_2)
    assert not torch.allclose(out, out_2)
    