import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from typing import List
from pusoy.action import Action

SOFTMAX_SIZES = [52, 5, 5]


def ppo_loss(
    curr_model: torch.nn.Module,
    prev_model: torch.nn.Module,
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
    curr_model.to(device)
    prev_model.to(device)
    actions = sum(actions, [])
    rewards = torch.cat(rewards, dim=0).reshape(-1, 1)
    lengths = torch.tensor([len(l) for l in inputs])
    cum_lengths = torch.cumsum(lengths, dim=0)

    input = pad_sequence(inputs, padding_value=0).to(device)
    input = pack_padded_sequence(input, lengths=lengths, enforce_sorted=False)
    output, state_values, _ = curr_model(input, packed=True)
    prev_output, _, _ = prev_model(input, compute_critic=False, packed=True)
    curr_log_probs = logits_to_log_probs(output, SOFTMAX_SIZES, device)
    prev_log_probs = logits_to_log_probs(prev_output, SOFTMAX_SIZES, device)
    ratios = torch.exp(curr_log_probs - prev_log_probs)

    batch_mask = batch_generate_mask(actions, device)
    ratios = ratios

    adv, critic_loss = gae(state_values, gamma, lambd, rewards, device, cum_lengths)
    surr1 = ratios * adv
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv

    surr_loss = torch.min(surr1, surr2) * batch_mask

    return (
        -surr_loss.mean()
        + c_critic * critic_loss
        - c_entropy * entropy(curr_log_probs, SOFTMAX_SIZES)
    )


def gae(values, gamma, lambd, rewards, device, cum_lengths):
    values_moved = torch.vstack((values[1:].detach(), torch.zeros((1, 1)).to(device)))
    values_moved[cum_lengths - 1] = 0
    target = rewards + (gamma * values_moved)
    td_estimates = target - values

    adv = torch.empty(td_estimates.shape).to(device)
    disc_factor = gamma * lambd
    start = 0
    for end in cum_lengths:
        end = end.item()
        len_seq = end - start
        matrix = torch.triu(
            (disc_factor ** -(torch.arange(len_seq))).reshape(-1, 1)
            * (disc_factor ** (torch.arange(len_seq))).reshape(1, -1)
        ).to(device)
        adv[start:end] = matrix @ td_estimates[start:end]
        start = end

    return adv, F.mse_loss(values, values + adv)


def td(values, gamma, lambd, rewards, device, cum_lengths):
    values_moved = torch.vstack((values[1:].detach(), torch.zeros((1, 1)).to(device)))
    values_moved[cum_lengths - 1] = 0
    target = rewards + (gamma * values_moved)
    td_estimates = target - values
    return td_estimates


def batch_generate_mask(actions, device):
    """Creates a mask for the output tensors using a list of Action objects."""
    card_tensors, round_tensors, hand_tensors = tuple(
        zip(
            *[
                (action.cards, action.type.to_tensor(), action.hand.to_tensor())
                for action in actions
            ]
        )
    )
    card_tensors, round_tensors, hand_tensors = (
        torch.stack(card_tensors),
        torch.stack(round_tensors),
        torch.stack(hand_tensors),
    )
    mask = torch.cat([card_tensors, round_tensors, hand_tensors], dim=1).to(device)
    return mask


def entropy(log_probs, softmax_sizes):
    """
    Calculates entropy of the output batch-wise given the log probabilities.
    """
    ptr = 0
    lst = []
    for size in softmax_sizes:
        selected = log_probs[:, ptr : ptr + size]
        probs = torch.exp(selected)
        entropy = -torch.sum(selected * probs, dim=-1)
        ptr += size
        lst.append(entropy)

    return torch.sum(torch.cat(lst, dim=-1), dim=-1)


def logits_to_log_probs(logits, softmax_sizes=[52, 5, 5], device="cuda"):
    """
    Converts input of (batch_size, sum(softmax_sizes)) logits into log probs.
    """
    ptr = 0
    lst = []
    for size in softmax_sizes:
        selected = logits[:, ptr : ptr + size]
        lst.append(F.log_softmax(selected, dim=-1))
        ptr += size
    return torch.cat(lst, dim=-1)
