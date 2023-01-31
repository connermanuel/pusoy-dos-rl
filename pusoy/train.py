from pusoy.game import Game
from pusoy.player import Player
from pusoy.models import D2RLModelWithCritic
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.losses import batch_loss_with_critic

import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool, Manager, cpu_count, set_start_method

import os
import argparse
import random
import joblib
import copy
from typing import Callable

def train(curr_model: torch.nn.Module, prev_model: torch.nn.Module, models: list, loss_func: Callable, 
          epochs: int=1500, batch_size: int=20, experience_replay_mult: int=4, device: str='cuda', 
          lr: float=0.001, num_models: int=15, eps: float=0.1, pool_size=4):
    """
    Trains curr_model using self-play.
    
    Parameters:
    curr_model -- the model to be trained.
    prev_model -- the model directly preceding the current model, used to evaluate PPO-clip loss.
    models -- a list of models to draw adversaries from.
    loss_func -- the function used to evaluate training loss.
    epochs -- training epochs
    batch_size -- number of games to play in each epoch
    experience_replay_mult -- sets experience replay size to (batch_size * experience replay mult)
    device -- device to perform operations on
    lr -- learning rate
    num_models -- number of models to store as adversaries
    eps -- epsilon for loss function clipping    
    """
    torch.autograd.set_detect_anomaly(True)
    experience_replay_size = batch_size * experience_replay_mult

    model_dir = "./models_{lr}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    curr_model.to(device)
    prev_model.to(device)
    opt = torch.optim.Adam(curr_model.parameters(), lr=lr)

    total_winning_actions = []
    total_losing_actions = []
    wins_list = []
    manager = Manager()

    print(f'Beginning training')
    print(f"Cpu count: {cpu_count()}")

    for epoch in range(1, epochs+1):
        wins = 0
        # print(f'Playing {batch_size} rounds')
        # for iter in range(batch_size):
        #     wins += play_round(curr_model, models, total_winning_actions, total_losing_actions, device, eps)
        # total_winning_actions[-batch_size:] = [[(t[0].cuda(), t[1]) for t in lst] for lst in total_winning_actions[-batch_size:]]
        # total_losing_actions[-batch_size:] = [[(t[0].cuda(), t[1]) for t in lst] for lst in total_losing_actions[-batch_size:]]

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
        
        # print(f'Selecting indices for training...')
        batch_number = -torch.log2(torch.rand(experience_replay_size))
        batch_number = torch.trunc(torch.clamp(batch_number, max=min(epoch-1, num_models-1)))
        unit_number = torch.randint(low=0, high=batch_size, size=(experience_replay_size,))
        idxs = (batch_number * batch_size) + unit_number
        idxs = idxs.to(torch.int)

        batch_winning_actions = sum([total_winning_actions[-(idx+1)] for idx in idxs], [])
        batch_losing_actions = sum([total_losing_actions[-(idx+1)] for idx in idxs], [])
        win_ratio = len(batch_winning_actions) / len(batch_losing_actions)

        # print(f'Training on winning actions...')
        inputs, actions = tuple(zip(*batch_winning_actions))
        loss = loss_func(curr_model, prev_model, inputs=inputs, actions=actions, adv=1, eps_clip=0.1, device=device)
        opt.zero_grad()
        try:
            loss.mean(dim=1).sum().backward()
        except RuntimeError as e:
            print(f'Loss: {loss} caused runtime error, skipping for now')
        opt.step()

        # print(f'Training on losing actions...')
        inputs, actions = tuple(zip(*batch_losing_actions))
        loss = loss_func(curr_model, prev_model, inputs=inputs, actions=actions, adv=-win_ratio, eps_clip=0.1, device=device)
        opt.zero_grad()
        try:
            loss.mean(dim=1).sum().backward()
        except RuntimeError as e:
            print(f'Loss: {loss} caused runtime error, skipping for now')
        opt.step()

        print(f'Epoch {epoch} winrate: {wins/batch_size}')

        if not epoch % 500:
            torch.save(curr_model.state_dict(), "{model_dir}/{epoch}.pt")
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
    adversaries = [TrainingDecisionFunction(model, device=device, eps=eps) for model in random.sample(list_of_models, 3)]
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
    model = D2RLModelWithCritic()
    old_model = copy.deepcopy(model)
    models = [old_model, old_model, old_model]
    model.load_state_dict(torch.load(f'models/500.pt'))
    
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

    
    args = parser.parse_args()

    train(model, old_model, models, loss_func=batch_loss_with_critic, 
          lr=args.lr, batch_size=args.batch_size, pool_size=args.pool_size, epochs=1000)
