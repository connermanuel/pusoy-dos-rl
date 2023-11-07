from typing import Callable
from torch import Tensor
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from pusoy.losses import batch_generate_mask, logits_to_log_probs, ppo_loss
from pusoy.action import PlayCards
from pusoy.utils import RoundType, Hands
from pusoy.models import A2CLSTM
from pusoy.decision_function import Neural, generate_input_from_state
from pusoy.train import (
    play_round_async,
    pool_callback,
    train_step,
    create_copy,
    ExperienceBuffer,
)
import pytest
import torch
import copy


# region FIXTURES
# ------------- FIXTURES -----------------
@pytest.fixture
def model():
    return A2CLSTM(hidden_dim=256)


@pytest.fixture
def model_copy(model: A2CLSTM) -> A2CLSTM:
    return create_copy(A2CLSTM, 256, model)


@pytest.fixture
def create_card_list():
    def _create_card_list(idxs):
        card_list = torch.zeros(52)
        for idx in idxs:
            card_list[idx] = 1
        return card_list

    return _create_card_list


@pytest.fixture
def create_input_state():
    def _create_input_state(card_list: Tensor):
        played_cards = [torch.zeros(52) for _ in range(4)]
        for i in range(1, 4):
            played_cards[i][i] = 1
        return generate_input_from_state(
            0,
            card_list,
            RoundType.SINGLES.to_tensor(dtype=torch.float),
            Hands.NONE.to_tensor(dtype=torch.float),
            played_cards[3],
            3,
            played_cards,
        )

    return _create_input_state


@pytest.fixture()
def create_action():
    def _create_action(card_list: Tensor):
        return PlayCards(cards=card_list, type=RoundType.SINGLES)

    return _create_action


@pytest.fixture()
def opt(model: A2CLSTM):
    return torch.optim.Adam(model.parameters(), lr=1e-4)


@pytest.fixture()
def buffer() -> ExperienceBuffer:
    return ExperienceBuffer()


# ----------------------------------------
# endregion


def test_loss_basic_pos(
    model: A2CLSTM,
    model_copy: A2CLSTM,
    create_input_state: Callable[..., Tensor],
    create_card_list: Callable[..., Tensor],
    create_action: Callable[..., PlayCards],
    opt: Adam,
):
    pos_action = create_action(create_card_list([3]))
    model = model.to("cuda")
    input_state = create_input_state(create_card_list([3, 4])).to("cuda").reshape(1, -1)
    actor_out_1, critic_out_1, _ = model(input_state)
    for _ in range(10):
        opt.zero_grad()
        loss = ppo_loss(
            model,
            model_copy,
            [input_state],
            [[pos_action]],
            [torch.tensor([10]).cuda()],
        )
        loss.backward()
        opt.step()
    actor_out_2, critic_out_2, _ = model(input_state)
    assert actor_out_2[0][3] > actor_out_1[0][3]
    assert critic_out_2 > critic_out_1


def test_loss_basic_neg(
    model: A2CLSTM,
    model_copy: A2CLSTM,
    create_input_state: Callable[..., Tensor],
    create_card_list: Callable[..., Tensor],
    create_action: Callable[..., PlayCards],
    opt: Adam,
):
    neg_action = create_action(create_card_list([4]))
    model = model.to("cuda")
    input_state = create_input_state(create_card_list([3, 4])).to("cuda").reshape(1, -1)
    actor_out_1, critic_out_1, _ = model(input_state)
    for _ in range(10):
        opt.zero_grad()
        loss = ppo_loss(
            model,
            model_copy,
            [input_state],
            [[neg_action]],
            [torch.tensor([-1]).cuda()],
        )
        loss.backward()
        opt.step()
    actor_out_2, critic_out_2, _ = model(input_state)
    assert actor_out_2[0][4] < actor_out_1[0][4]
    assert critic_out_2 < critic_out_1


def test_loss_basic(
    model: A2CLSTM,
    model_copy: A2CLSTM,
    create_input_state: Callable[..., Tensor],
    create_card_list: Callable[..., Tensor],
    create_action: Callable[..., PlayCards],
    opt: Adam,
):
    pos_action = create_action(create_card_list([3]))
    neg_action = create_action(create_card_list([4]))
    model = model.to("cuda")
    input_state = create_input_state(create_card_list([3, 4])).to("cuda").reshape(1, -1)
    actor_out_1, _, _ = model(input_state)
    for _ in range(10):
        opt.zero_grad()
        loss = ppo_loss(
            model,
            model_copy,
            [input_state, input_state],
            [[pos_action], [neg_action]],
            [torch.tensor([1]).cuda(), torch.tensor([-1]).cuda()],
        )
        loss.backward()
        opt.step()
    actor_out_2, _, _ = model(input_state)
    assert actor_out_2[0][3] > actor_out_1[0][3]
    assert actor_out_2[0][4] < actor_out_1[0][4]


def test_loss_game(
    model: A2CLSTM,
    model_copy: A2CLSTM,
    opt: Adam,
    buffer: ExperienceBuffer,
):
    res = play_round_async([model, model_copy, model_copy, model_copy], [])
    pool_callback(res, buffer)

    buffer.win_inputs = [l[0:4] for l in buffer.win_inputs]
    buffer.win_actions = [l[0:4] for l in buffer.win_actions]
    buffer.lose_inputs = [l[0:4] for l in buffer.lose_inputs]
    buffer.lose_actions = [l[0:4] for l in buffer.lose_actions]

    win_inputs = buffer.win_inputs[0].to("cuda")
    win_actions = buffer.win_actions[0]
    lose_inputs = pack_padded_sequence(
        pad_sequence(buffer.lose_inputs),
        lengths=[len(l) for l in buffer.lose_inputs],
        enforce_sorted=False,
    ).to("cuda")
    lose_actions = sum(buffer.lose_actions, [])

    win_mask = batch_generate_mask(win_actions, "cuda")
    lose_mask = batch_generate_mask(lose_actions, "cuda")

    win_actor_out_1, win_critic_out_1, _ = model(win_inputs)
    lose_actor_out_1, lose_critic_out_1, _ = model(lose_inputs, packed=True)
    for _ in range(100):
        train_step(model, model_copy, opt, buffer, device="cuda", c_entropy=0)
        model_copy = create_copy(A2CLSTM, 256, model)
    win_actor_out_2, win_critic_out_2, _ = model(win_inputs)
    lose_actor_out_2, lose_critic_out_2, _ = model(lose_inputs, packed=True)

    win_actor_diff = (
        logits_to_log_probs(win_actor_out_2) -
        logits_to_log_probs(win_actor_out_1)
    ) * win_mask
    lose_actor_diff = (
        logits_to_log_probs(lose_actor_out_2) -
        logits_to_log_probs(lose_actor_out_1)
    ) * lose_mask
    w_a = win_actor_diff.flatten()
    w_a = w_a[w_a != 0]
    l_a = lose_actor_diff.flatten()
    l_a = l_a[l_a != 0]

    win_critic_diff = (win_critic_out_2 - win_critic_out_1)
    lose_critic_diff = (lose_critic_out_2 - lose_critic_out_1)
    assert torch.all(win_actor_diff >= 0)
    return
