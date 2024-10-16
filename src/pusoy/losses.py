import torch
import torch.nn.functional as F

from pusoy.action import Action
from pusoy.constants import DEVICE, OUTPUT_SIZES
from pusoy.models import PusoyModel


def ppo_loss(
    curr_model: PusoyModel,
    prev_model: PusoyModel,
    inputs: torch.Tensor,
    adv: torch.Tensor,
    batch_masks: tuple[torch.Tensor],
    eps_clip: float = 0.1,
    gamma: float = 0.99,
    lambd: float = 0.9,
    c_entropy: float = 0.001,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Returns negated reward (positive action should have negative loss) of a batch of
    inputs and actions. For a feed forward model we assume conditional independence.

    The inputs and model should already be on the correct device.

    Args:
        curr_model: the current model
        prev_model: the previous model
        inputs: (batch_size, input_dim) the input to the model representing
        the game state
        adv: (batch_size) tensor of advantages at each timestep. The most basic version
        is simply a 1 or -1/3 at the end of each game.
        batch_masks: A tuple of three (batch_size, _) tensors that mask out
            the cards, hands, and round
        eps_clip: epsilon for clipping gradient update
        gamma: discount factor for computing advantage estimate
        c_entropy: weight of entropy term (higher alpha = more exploration)
        device: device to perform operations on
    
    Returns:
        A tensor corresponding to the PPO loss.
    """
    output, state_values = curr_model.forward(inputs)
    prev_output = prev_model.act(inputs)

    log_probs: list[torch.Tensor] = [F.log_softmax(t, dim=-1) for t in output]
    prev_log_probs: list[torch.Tensor] = [F.log_softmax(t, dim=-1) for t in prev_output]
    ratios = [torch.exp(t - prev_t) for t, prev_t in zip(log_probs, prev_log_probs)]
    ratios = [r * b for r, b in zip(ratios, batch_masks)]

    critic_loss = F.mse_loss(state_values, state_values + adv)
    surr1 = ratios * adv
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv

    objective = (
        torch.min(surr1, surr2).mean()
        - critic_loss
        + c_entropy * entropy(log_probs, OUTPUT_SIZES)
    )

    return -objective


def gae(
    state_values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.99,
    lambd: float = 0.9,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Computes the generalized advantage estimate over time for a specific

    Args:
        state_values: (seq_len,) tensor of state values
        rewards: (seq_len,) a reward tensor corresponding to reward after each action
            inside a game sequence
        gamma:
        lambd:
        device:
    """
    values_moved = torch.stack((state_values[1:].detach(), torch.zeros(1).to(device)))
    target = rewards + (gamma * values_moved)
    td_estimates = target - state_values

    disc_factor = gamma * lambd
    end = len(state_values)
    matrix = torch.triu(
        (disc_factor ** -(torch.arange(end))).reshape(-1, 1)
        * (disc_factor ** (torch.arange(end))).reshape(1, -1)
    ).to(device)
    adv = matrix @ td_estimates

    return adv


def td(values, gamma, lambd, rewards, device, cum_lengths):
    values_moved = torch.vstack((values[1:].detach(), torch.zeros((1, 1)).to(device)))
    values_moved[cum_lengths - 1] = 0
    target = rewards + (gamma * values_moved)
    td_estimates = target - values
    return td_estimates


def batch_generate_mask(actions: list[Action], device) -> torch.Tensor:
    """Creates a mask for the output tensors using a list of Action objects.

    Args:
        actions: A list of actions corresponding to a played game.
        device: The device on which to place the batch mask.

    Returns:
        A boolean tensor with shape (output_dim,) that masks out the relevant logits for the selected action.
    """
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

    TO GET THE ENTROPY:
    - https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py#L104C9-L119C42
    - entropy dist from constant var matrix and predicted action vectors
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
