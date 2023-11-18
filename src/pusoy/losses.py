import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from typing import List
from pusoy.action import Action
from pusoy.models import PusoyModel

SOFTMAX_SIZES = [52, 5, 5]

def ppo_loss(
    curr_model: PusoyModel,
    prev_model: PusoyModel,
    inputs: List[torch.tensor],
    actions: List[List[Action]],
    rewards: List[torch.Tensor],
    eps_clip: float = 0.1,
    gamma: float = 0.99,
    lambd: float = 0.9,
    c_critic: float = 1,
    c_entropy: float = 0.001,
    device: str = "cuda",
):
    """
    Returns negated reward (positive action should have negative loss) of a batch of inputs and actions.
    Each batch corresponds to a nested list of sequences, each sequence representing one player in a game.
    Args:
    - curr_model: Model -- the current model
    - prev_model: Model -- the previous model
    - inputs: list[torch.tensor] -- (batch_size, (seq_len, input_dim)) list of tensors, each representing a game
    - actions: list[Action] -- (batch_size, seq_len) list of lists of actions taken by the model using the output
    - rewards: torch.tensor -- (batch_size, seq_len) list of reward tensors corresponding to each input + action
    - eps_clip: float -- epsilon for clipping gradient update
    - gamma: float -- discount factor
    - c_entropy: float -- weight of entropy term (higher alpha = more exploration)
    - device: str -- device to perform operations on
    """
    surr_loss_total = 0
    critic_loss_total = 0
    entropy_total = 0
    curr_model.to(device)
    prev_model.to(device)

    for input, action, reward in zip(inputs, actions, rewards):
        curr_output = curr_model.get_ppo_output_vector(input, action)
        prev_output = prev_model.get_ppo_output_vector(input, action)
        ratios = curr_output["probs"] / prev_output["probs"]
        mask = curr_output["mask"]

        adv, critic_loss = gae(curr_output["value"], gamma, lambd, reward, device)
        adv = adv[mask.nonzero(as_tuple=True)[0]]
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv

        surr_loss = torch.min(surr1, surr2).sum(dim=1).mean()
        entropy = - (curr_output["probs"] * torch.log(curr_output["probs"])).sum()

        surr_loss_total += surr_loss
        critic_loss_total += critic_loss
        entropy_total += entropy

    return -surr_loss_total + c_critic * critic_loss_total - c_entropy * entropy_total


def gae(values, gamma, lambd, rewards, device):
    """
    Computes a generalized advantage estimate, using the model's value function.
    Args:
    - values: (S) tensor of state values
    - rewards: (S) tensor of state rewards
    """
    values_moved = torch.cat((values[1:].detach(), torch.zeros(1).to(device)))
    target = rewards + (gamma * values_moved)
    td_estimates = target - values

    disc_factor = gamma * lambd
    matrix = torch.triu(
        (disc_factor ** -(torch.arange(values.shape[0]))).reshape(-1, 1)
        * (disc_factor ** (torch.arange(values.shape[0]))).reshape(1, -1)
    ).to(device)
    adv = matrix @ td_estimates

    return adv.reshape(-1, 1), F.mse_loss(values, values + adv)
