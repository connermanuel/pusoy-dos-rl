from pusoy.game import Game
from pusoy.player import Player
from pusoy.models import D2RL, D2RLAC, D2RLA2C, D2RLAQC, D2RLA2QC
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.losses import ppo_loss

import torch
from torch.multiprocessing import Pool, Manager, cpu_count, set_start_method

import os
import argparse
import random
import joblib
import copy

def train(curr_model: torch.nn.Module, num_models: int=15, epochs: int=1500, batch_size: int=20, 
          experience_replay_mult: int=4, lr_actor: float=0.001, lr_critic: float=0.001, eps: float=0.1, 
          gamma: float=0.99, alpha: float=0.001, pool_size: int=4, save_steps: int=500, device: str='cuda', model_dir=None):
    """
    Trains curr_model using self-play.

    Parameters:
    curr_model -- the model to be trained.
    loss_func -- the function used to evaluate training loss.
    num_models -- number of models to store as adversaries
    epochs -- training epochs
    batch_size -- number of games to play in each epoch
    experience_replay_mult -- sets experience replay size to (batch_size * experience replay mult)
    lr_actor -- learning rate for actor
    lr_critic -- learning rate for critic
    eps -- epsilon for loss function clipping    
    gamma -- discount factor for rewards
    alpha -- entropy factor
    pool_size -- number of processes to spawn
    save_steps -- number of steps to take before saving
    device -- device to perform operations on
    model_dir -- where to save the models
    """
    torch.autograd.set_detect_anomaly(True)
    experience_replay_size = batch_size * experience_replay_mult

    prev_model = copy.deepcopy(curr_model)
    models = [prev_model]
    curr_model.to(device)
    prev_model.to(device)

    if model_dir is None:
        model_dir = f"./models_a{lr_actor}_c{lr_critic}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if curr_model.critic:
        opt = torch.optim.Adam([
            {'params': curr_model.actor.parameters(), 'lr': lr_actor},
            {'params': curr_model.critic.parameters(), 'lr': lr_critic}
        ])
    else:
        opt = torch.optim.Adam(curr_model.parameters(), lr_actor)

    total_winning_actions = []
    total_losing_actions = []
    wins_list = []
    manager = Manager()

    print(f'Beginning training')
    print(f"Cpu count: {cpu_count()}")

    for epoch in range(1, epochs+1):
        wins = 0
        
        winning_actions, losing_actions = manager.list(), manager.list()
        with Pool(pool_size) as pool:
            args = [curr_model, models, winning_actions, losing_actions, device, eps]
            wins = [pool.apply_async(play_round, args=args) for i in range(batch_size)]
            wins = [res.get(timeout=60) for res in wins]
            wins = torch.sum(torch.tensor(wins)).item()

        for i in range(batch_size):
            total_winning_actions.append([(t[0].clone().cuda(), t[1]) for t in winning_actions[i]])
            total_losing_actions.append([(t[0].clone().cuda(), t[1]) for t in losing_actions[i]])
        del winning_actions, losing_actions
        
        # print('Selecting indices for training...')
        batch_number = -torch.log2(torch.rand(experience_replay_size))
        batch_number = torch.trunc(torch.clamp(batch_number, max=min(epoch-1, num_models-1)))
        unit_number = torch.randint(low=0, high=batch_size, size=(experience_replay_size,))
        idxs = (batch_number * batch_size) + unit_number
        idxs = idxs.to(torch.int)

        # Nested list of winning and losing actions
        batch_winning_actions = [total_winning_actions[-(idx+1)] for idx in idxs]
        batch_losing_actions = [total_losing_actions[-(idx+1)] for idx in idxs]
        # Normalize rewards
        winning_actions_rewards = [gamma ** torch.arange(len(l), device=device) for l in batch_winning_actions]
        losing_actions_rewards = [gamma ** torch.arange(len(l), device=device) for l in batch_losing_actions]
        winning_actions_rewards_sum = torch.sum(torch.concat(winning_actions_rewards))
        losing_actions_rewards_sum = torch.sum(torch.concat(losing_actions_rewards))
        win_ratio = winning_actions_rewards_sum / losing_actions_rewards_sum
        losing_actions_rewards = [t * -win_ratio for t in losing_actions_rewards]
        
        win_inputs, win_actions = tuple(zip(*sum(batch_winning_actions, [])))
        lose_inputs, lose_actions = tuple(zip(*sum(batch_losing_actions, [])))
        win_rewards = torch.concat(winning_actions_rewards)
        lose_rewards = torch.concat(losing_actions_rewards)

        inputs = win_inputs + lose_inputs
        actions = win_actions + lose_actions
        rewards = torch.concat([win_rewards, lose_rewards])

        # print(f'Training on winning actions...')
        loss = ppo_loss(curr_model, prev_model, inputs, actions, rewards, eps_clip=eps, 
                        gamma=gamma, alpha=alpha, device=device)
        opt.zero_grad()
        try:
            loss.backward()
            opt.step()
        except RuntimeError as e:
            print(f'Loss: {loss} caused runtime error, skipping for now')

        print(f'Epoch {epoch} winrate: {wins/batch_size}')

        if not epoch % save_steps:
            torch.save(curr_model.state_dict(), f"{model_dir}/{epoch}.pt")
        prev_model.load_state_dict(curr_model.state_dict())
        models.append(prev_model)
        if len(models) > num_models:
            del models[0]
        if len(total_winning_actions) > num_models * batch_size:
            del total_winning_actions[:batch_size]
            del total_losing_actions[:batch_size]

    joblib.dump(wins_list, 'wins_list.pkl')

    return wins_list

def play_round(
        curr_model: torch.nn.Module, 
        list_of_models: list[torch.nn.Module], 
        winning_actions: list[tuple], 
        losing_actions: list[tuple],
        device='cuda',
        eps=0.1
    ):
    """
    Plays a round, and appends the winning actions and losing actions to the existing lists of actions.
    Args:
    - curr_model: Model -- the model currently being trained and evaluated
    - list_of_models: list[Model] -- a list of previously trained models to be selected from as adversaries
    - winning_actions: list[tuple] -- a list of (input, action) pairs from winning players
    - losing_actions: list[tuple] -- a list of (input, action) pairs from losing players
    """
    player = TrainingDecisionFunction(curr_model, device=device, eps=eps)
    adversaries = [TrainingDecisionFunction(model, device=device, eps=eps) for model in random.choices(list_of_models, k=3)]
    players = [player] + adversaries
    players = [Player(None, i, p) for i, p in zip(range(4), players)]

    game = Game(players)
    game.play()

    for player in players:
        instances = [(t[0].cpu(), t[1]) for t in player.decision_function.instances]
        if player.winner:
            winning_actions.append(instances)
        else:
            losing_actions.append(instances)

    return player.winner


if __name__ == "__main__":
    set_start_method("spawn")
    # model.load_state_dict(torch.load(f'models/500.pt')) 
    parser = argparse.ArgumentParser(description="Train PUSOY model.")
    
    parser.add_argument("-p", "--pool_size", 
        help="Number of CPU processes to spawn. Defaults to 2.",
        type=int, default=2)
    parser.add_argument("-b", "--batch_size", 
        help="Batch size. Defaults to 20.",
        type=int, default=20)
    parser.add_argument("--lr", 
        help="Learning rate. Defaults to 1e-03 (0.001).",
        type=float, default=1e-03)
    parser.add_argument("--er_mult",
        help="Experience replay mult. Defaults to 4.",
        type=int, default=4)
    parser.add_argument("-m", "--model",
        help="Model architecture to train. Defaults to A2C.",
        options=["base", "ac", "a2c", "aqc", "a2qc"],
        type=str, default="a2c")
    parser.add_argument("-m", "--model",
        help="Model architecture to train. Defaults to A2C.",
        options=["base", "ac", "a2c", "aqc", "a2qc"],
        type=str, default="a2c")
    
    model_dispatch = {
        "base": D2RL,
        "ac": D2RLAC,
        "a2c": D2RLA2C,
        "aqc": D2RLAQC,
        "a2qc": D2RLA2QC
    }

    args = parser.parse_args()
    ModelClass = args.model
    model = ModelClass()

    train(model, lr=args.lr, batch_size=args.batch_size, pool_size=args.pool_size, epochs=1000)
