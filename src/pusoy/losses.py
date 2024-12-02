import torch
import torch.nn.functional as F

from pusoy.action import Action
from pusoy.constants import DEVICE, OUTPUT_SIZES
from pusoy.models import PusoyModel


def ppo_loss(
    curr_log_probs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    prev_log_probs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state_values: torch.Tensor,
    advantages: torch.Tensor,
    action_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps_clip: float = 0.2,
    c_entropy: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute PPO loss with clipped objective for multi-part pusoy action space.

    Args:
        curr_log_probs: Tuple of (cards, hand_type, round_type) log probs
        prev_log_probs: Tuple of old policy log probs
        state_values: (batch_size,) Value predictions
        advantages: (batch_size,) Advantage estimates
        action_masks: Tuple of masks for each action part
        eps_clip: PPO clip parameter
        c_entropy: Entropy bonus coefficient
    """
    # Input validation
    assert len(curr_log_probs) == len(prev_log_probs) == len(action_masks) == 3
    assert all(
        c.shape[0] == p.shape[0] == advantages.shape[0]
        for c, p in zip(curr_log_probs, prev_log_probs)
    )

    actor_losses = []
    approx_kls = []

    # Compute losses per action part
    for curr_probs, prev_probs, mask in zip(
        curr_log_probs, prev_log_probs, action_masks
    ):
        log_ratio = curr_probs - prev_probs
        ratio = torch.exp(log_ratio) * mask

        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages.unsqueeze(
            -1
        )

        actor_losses.append(-torch.min(surr1, surr2).mean())
        approx_kls.append(((ratio - 1) - log_ratio).mean().item())

    # Combine losses
    actor_loss = sum(actor_losses)
    critic_loss = 0.5 * F.mse_loss(state_values, state_values + advantages)
    entropy_loss = -c_entropy * sum(entropy(probs) for probs in curr_log_probs)

    total_loss = actor_loss + critic_loss + entropy_loss

    metrics = {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "approx_kl": sum(approx_kls) / len(approx_kls),
    }

    return total_loss, metrics


def split_and_compute_log_probs(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split 62-dim logits tensor into three parts and compute log probabilities.
    Concantenates the log probabilities of each part into a single tensor.

    Args:
        logits: (batch_size, 62) tensor

    Returns:
        Tuple of log probabilities for each part:
        - (batch_size, 52) for first part
        - (batch_size, 5) for second part
        - (batch_size, 5) for third part
    """
    first_logits = logits[..., :52]
    second_logits = logits[..., 52:57]
    third_logits = logits[..., 57:]

    first_log_probs = F.log_softmax(first_logits, dim=-1)
    second_log_probs = F.log_softmax(second_logits, dim=-1)
    third_log_probs = F.log_softmax(third_logits, dim=-1)

    return first_log_probs, second_log_probs, third_log_probs


def gae(
    state_values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.99,
    lambd: float = 0.9,
) -> torch.Tensor:
    """
    Computes the generalized advantage estimate over time for a player.
    From the paper High-Dimensional Continuous Control Using Generalized Advantage Estimation
    https://arxiv.org/abs/1506.02438

    Args:
        state_values: (seq_len,) tensor of state values
        rewards: (seq_len,) a reward tensor corresponding to reward after each action
            inside a game sequence
        gamma: Discount factor for future rewards
        lambd: Trace decay factor
        device: Device to perform operations on
    """
    advantages = torch.empty(rewards.size(), device=rewards.device)
    advantage = 0
    next_value = 0

    for i, (r, v) in reversed(list(enumerate(zip(rewards, state_values)))):
        td_error = r + next_value * gamma - v
        advantage = td_error + advantage * gamma * lambd
        next_value = v
        advantages[i] = advantage

    return advantages


def td(
    state_values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.99,
    lambd: float = 0.9,
) -> torch.Tensor:
    """
    Computes the temporal difference estimate over time for a player.

    Args:
        state_values: (seq_len,) tensor of state values
        rewards: (seq_len,) a reward tensor corresponding to reward after each action
            inside a game sequence
        gamma: Discount factor for future rewards
        lambd: Trace decay factor. Unused in TD.
        device: Device to perform operations on
    """
    advantages = []
    next_value = 0

    for r, v in zip(reversed(rewards), reversed(state_values)):
        advantage = r + next_value * gamma - v
        next_value = v
        advantages.insert(0, advantage)

    return torch.tensor(advantages)


def generate_action_masks(actions: list[Action]) -> torch.Tensor:
    """Creates a mask for the output tensors using a list of Action objects.

    Args:
        actions: A list of actions corresponding to a played game.

    Returns:
        Tuple of (cards_mask, hand_type_mask, round_type_mask)
    """
    card_tensors = []
    round_tensors = []
    hand_tensors = []

    for action in actions:
        card_tensors.append(action.cards)

        round_tensor = action.type.to_tensor(dtype=torch.float)
        round_tensors.append(round_tensor)

        hand_tensor = action.hand.to_tensor(dtype=torch.float)
        hand_tensors.append(hand_tensor)

    card_mask = torch.stack(card_tensors, dim=0)
    round_mask = torch.stack(round_tensors, dim=0)
    hand_mask = torch.stack(hand_tensors, dim=0)

    return card_mask, round_mask, hand_mask


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy given log probabilities.

    Args:
        log_probs: (batch_size, action_dim) tensor of log probabilities

    Returns:
        (batch_size,) tensor of entropy values
    """
    probs = torch.exp(log_probs)
    return -torch.sum(log_probs * probs, dim=-1)
