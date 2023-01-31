import torch
import torch.nn.functional as F

def batch_loss(curr_model, prev_model, inputs, actions, adv, eps_clip=0.1, device='cuda'):
    """
    Returns negated loss (positive action should have negative loss) of a batch of inputs and actions.
    Args:
    - curr_model: Model -- the current model
    - prev_model: Model -- the previous model
    - inputs: list[torch.tensor] -- (batch_size, input_dim) the input to the model representing the game state
    - actions: list[Action] -- (batch_size, action_dim) list of actions taken by the model using the output
    - adv: int -- 1 if the model won the game, and -1/3 if the model lost
    """
    softmax_sizes = [52, 5, 5]
    input = torch.stack(inputs)
    output = curr_model(input)
    prev_output = prev_model(input).detach()
    curr_log_probs = logits_to_log_probs(output, softmax_sizes, device)
    prev_log_probs = logits_to_log_probs(prev_output, softmax_sizes, device)
    
    ratios = torch.exp(curr_log_probs - prev_log_probs)

    batch_mask = batch_generate_mask(actions)
    ratios = ratios * batch_mask.to(device)

    surr1 = ratios * adv
    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * adv

    return -torch.min(surr1, surr2) - 0.01 * entropy(curr_log_probs, softmax_sizes)

def batch_loss_with_critic(curr_model, prev_model, inputs, actions, adv, eps_clip=0.1, gamma=0.2, device='cuda'):
    """
    Returns negated loss (positive action should have negative loss) of a batch of inputs and actions.
    Args:
    - curr_model: Model -- the current model
    - prev_model: Model -- the previous model
    - inputs: list[torch.tensor] -- (batch_size, input_dim) the input to the model representing the game state
    - actions: list[Action] -- (batch_size, action_dim) list of actions taken by the model using the output
    - adv: int -- 1 if the model won the game, and -1/3 if the model lost
    """
    softmax_sizes = [52, 5, 5]
    input = torch.stack(inputs)
    output = curr_model(input)
    prev_output = prev_model(input).detach()
    curr_log_probs = logits_to_log_probs(output, softmax_sizes, device)
    prev_log_probs = logits_to_log_probs(prev_output, softmax_sizes, device)
    
    ratios = torch.exp(curr_log_probs - prev_log_probs)

    batch_mask = batch_generate_mask(actions)
    ratios = ratios * batch_mask.to(device)

    state_values = curr_model.critic(input)
    state_values_moved = torch.vstack((state_values[1:].detach(), torch.zeros((1, 1)).to(device)))
    adv = adv + (gamma * state_values_moved) - state_values

    surr1 = ratios * adv # Reward
    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * adv

    return -torch.min(surr1, surr2) - 0.5 * curr_model.MSELoss(adv, state_values) - 0.01 * entropy(curr_log_probs, softmax_sizes)

def batch_loss_with_q_value_critic(curr_model, prev_model, inputs, actions, adv, eps_clip=0.1, device='cuda'):
    """
    Returns negated loss (positive action should have negative loss) of a batch of inputs and actions.
    Args:
    - curr_model: Model -- the current model
    - prev_model: Model -- the previous model
    - inputs: list[torch.tensor] -- (batch_size, input_dim) the input to the model representing the game state
    - actions: list[Action] -- (batch_size, action_dim) list of actions taken by the model using the output
    - adv: int -- 1 if the model won the game, and -1/3 if the model lost
    """
    softmax_sizes = [52, 5, 5]
    input = torch.stack(inputs)
    output = curr_model(input)
    prev_output = prev_model(input).detach()
    curr_log_probs = logits_to_log_probs(output, softmax_sizes, device)
    prev_log_probs = logits_to_log_probs(prev_output, softmax_sizes, device)
    
    ratios = torch.exp(curr_log_probs - prev_log_probs)

    batch_mask = batch_generate_mask(actions).to(device)
    ratios = ratios * batch_mask

    q_vals = curr_model.critic(torch.hstack((input, batch_mask)))

    surr1 = ratios * q_vals # Reward
    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * adv
    adv = (torch.ones(q_vals.shape) * adv).to(device)

    return -torch.min(surr1, surr2) - 1.5 * curr_model.MSELoss(adv, q_vals) - 0.01 * entropy(curr_log_probs, softmax_sizes)

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
