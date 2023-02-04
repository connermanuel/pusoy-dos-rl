import torch
import torch.nn.functional as F
from typing import List
from action import Action

SOFTMAX_SIZES = [52, 5, 5]

def ppo_loss(curr_model: torch.nn.Module, prev_model: torch.nn.Module, inputs: List[torch.tensor], 
            actions: List[Action], rewards: torch.tensor, eps_clip: float=0.1, 
            gamma: float=0.99, alpha: float=0.001, device: str='cuda'):
    """
    Returns negated reward (positive action should have negative loss) of a batch of inputs and actions.
    Args:
    - curr_model: Model -- the current model
    - prev_model: Model -- the previous model
    - inputs: list[torch.tensor] -- (batch_size, input_dim) the input to the model representing the game state
    - actions: list[Action] -- list of actions taken by the model using the output
    - rewards: torch.tensor -- tensor of rewards corresponding to each input + action
    - eps_clip: float -- epsilon for clipping gradient update
    - gamma: float -- discount factor
    - alpha: float -- weight of entropy term (higher alpha = more exploration)
    - device: str -- device to perform operations on
    """
    input = torch.stack(inputs)
    output = curr_model.actor(input)
    prev_output = prev_model.actor(input).detach()
    curr_log_probs = logits_to_log_probs(output, SOFTMAX_SIZES, device)
    prev_log_probs = logits_to_log_probs(prev_output, SOFTMAX_SIZES, device)

    ratios = torch.exp(curr_log_probs - prev_log_probs)
    batch_mask = batch_generate_mask(actions)
    ratios = ratios * batch_mask.to(device)

    adv, critic_loss = curr_model.adv_func(curr_model, input, batch_mask, gamma, rewards, device)
    surr1 = ratios * adv
    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * adv

    return -torch.min(surr1, surr2).mean(dim=1).sum() + critic_loss - alpha * entropy(curr_log_probs, SOFTMAX_SIZES)

### Advantage Functions
def identity(curr_model, input, batch_mask, gamma, rewards, device):
    return rewards, 0

def state_value(curr_model, input, batch_mask, gamma, rewards, device):
    state_values = curr_model.critic(input)
    return state_values, F.mse_loss(state_values, rewards)

def q_value(curr_model, input, batch_mask, gamma, rewards, device):
    q_values = curr_model.critic(torch.hstack((input, batch_mask)))
    return q_values, F.mse_loss(q_values, rewards)

def state_value_advantage(curr_model, input, batch_mask, gamma, rewards, device):
    state_values = curr_model.critic(input)
    state_values_moved = torch.vstack((state_values[1:].detach(), torch.zeros((1, 1)).to(device)))
    target = rewards[..., None] + (gamma * state_values_moved)
    adv = target - state_values
    return adv, F.mse_loss(state_values, target)

def q_value_advantage(curr_model, input, batch_mask, gamma, rewards, device):
    q_values = curr_model.critic(torch.hstack((input, batch_mask)))
    q_values_moved = torch.vstack((q_values[1:].detach(), torch.zeros((1, 1)).to(device)))
    target = rewards + (gamma * q_values_moved)
    adv = target - q_values
    return adv, F.mse_loss(q_values, target)

# actor_only: use identity loss
# state_value_critic: use state_value
# q_value_critic: use either q value function

def batch_generate_mask(actions):
    """Creates a mask for the output tensors using a list of Action objects."""
    card_tensors, round_tensors, hand_tensors = tuple(zip(*[(action.cards, action.type.to_tensor(), action.hand.to_tensor()) for action in actions]))
    card_tensors, round_tensors, hand_tensors = torch.stack(card_tensors), torch.stack(round_tensors), torch.stack(hand_tensors)
    mask = torch.cat([card_tensors, round_tensors, hand_tensors], dim=1)
    return mask

def entropy(log_probs, softmax_sizes):
    """
    Calculates entropy of the output batch-wise given the log probabilities.
    """
    ptr = 0
    lst = []
    for size in softmax_sizes:
        selected = log_probs[:, ptr:ptr+size]
        probs = torch.exp(selected)
        entropy = -torch.sum(selected * probs, dim=-1)
        lst.append(entropy)
    
    return torch.sum(torch.cat(lst, dim=-1), dim=-1)

def logits_to_log_probs(logits, softmax_sizes=[52, 5, 5], device='cuda'):
    """
    Converts input of (batch_size, sum(softmax_sizes)) logits into log probs.
    """
    ptr = 0
    lst = []
    for size in softmax_sizes:
        selected = logits[:, ptr:ptr+size]
        lst.append(F.log_softmax(selected, dim=-1))
        ptr += size
    return torch.cat(lst, dim=-1)
