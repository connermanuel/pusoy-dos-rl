from pusoy.game import Game
from pusoy.player import Player
from pusoy.models import Base, D2RLAC, D2RLA2C, D2RLAQC, D2RLA2QC
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.losses import ppo_loss

import torch
from torch.multiprocessing import Process, Event, Queue, Manager, Pool, cpu_count, set_start_method, set_sharing_strategy
from queue import Empty

import os
import argparse
import random
import joblib
import copy
import time
import sys

from typing import List

def train(curr_model: torch.nn.Module, num_models: int=15, epochs: int=1500, batch_size: int=20, 
          experience_replay_mult: int=4, method: str="process", lr_actor: float=0.001, lr_critic: float=0.001, eps: float=0.1, 
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
    method -- whether to use process or pool
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
        os.makedirs(model_dir)
    
    if curr_model.critic:
        opt = torch.optim.Adam([
            {'params': curr_model.actor.parameters(), 'lr': lr_actor},
            {'params': curr_model.critic.parameters(), 'lr': lr_critic}
        ])
    else:
        opt = torch.optim.Adam(curr_model.parameters(), lr_actor)

    total_winning_actions, total_losing_actions = [], []

    m = Manager()
    queue = Queue()
    wins_list = []

    print(f'Beginning training')
    print(f"Cpu count: {cpu_count()}")

    for epoch in range(1, epochs+1):
        wins = 0

        if method == "process":
            ### Process implementation
            num_done = 0

            events = [Event() for i in range(pool_size)]
            init_args = [[curr_model, models, queue, events[id], id, device, eps] for id in range(pool_size)]
            processes = [Process(target=play_round, args=init_args[i]) for i in range(pool_size)]
            [process.start() for process in processes]
            
            while num_done < batch_size:
                try:
                    id, res, win_actions_orig, lose_actions_orig = queue.get(timeout=30)
                except Empty:
                    print("Timeout, resetting processes...")
                    [process.kill() for process in processes]
                    [process.join() for process in processes]
                    events.clear(), processes.clear()
                    events = [Event() for i in range(pool_size)]
                    init_args = [[curr_model, models, queue, events[id], id, device, eps] for id in range(pool_size)]
                    processes = [Process(target=play_round, args=init_args[i]) for i in range(pool_size)]
                    [process.start() for process in processes]
                    continue

                win_actions = [(t[0].clone(), t[1].clone()) for t in win_actions_orig]
                lose_actions = [(t[0].clone(), t[1].clone()) for t in lose_actions_orig]
                total_winning_actions.append(win_actions)
                total_losing_actions.append(lose_actions)
                del win_actions_orig, lose_actions_orig

                events[id].set()
                processes[id].join(5)
                if num_done < batch_size - pool_size:
                    events[id] = Event()
                    args = [curr_model, models, queue, events[id], id, device, eps]
                    process = Process(target=play_round, args=args)
                    process.start()
                    processes[id] = process

                num_done += 1
                wins += res
            [event.set() for event in events]
            [process.join() for process in processes]

        else:
            ### Pool implementation
            winning_actions, losing_actions = m.Queue(), m.Queue()
            args_async = [curr_model, models, winning_actions, losing_actions, device, eps]
            with Pool(pool_size, maxtasksperchild=1) as pool:
                wins = [pool.apply(play_round_async, args=args_async) for i in range(batch_size)]
                wins = torch.sum(torch.tensor(wins)).item()

            for i in range(batch_size):
                total_winning_actions.append([(t[0].to(device), t[1].to(device)) for t in winning_actions.get()])
                total_losing_actions.append([(t[0].to(device), t[1].to(device)) for t in losing_actions.get()])
#        
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
        curr_model: torch.nn.Module, list_of_models: List[torch.nn.Module], queue: Queue, 
        done_event: Event, id: int, device: str='cuda', eps: float=0.1,):
    """
    Plays a round, and appends the winning actions and losing actions to the existing lists of actions.
    - curr_model -- the model currently being trained and evaluated
    - list_of_models -- a list of previously trained models to be selected from as adversaries
    - winning_actions -- a list of (input, action) pairs from winning players
    - losing_actions -- a list of (input, action) pairs from losing players
    - done_queue -- queue to put message on when finished running
    - done_event -- event to wait for to end
    - device -- device to run everything on
    - eps -- epsilon for clipping gradients,
    - id -- the process id
    """
    player = TrainingDecisionFunction(curr_model, device=device, eps=eps)
    adversaries = [TrainingDecisionFunction(model, device=device, eps=eps) for model in random.choices(list_of_models, k=3)]
    players = [player] + adversaries
    players = [Player(i, p) for i, p in zip(range(4), players)]

    game = Game(players)
    game.play()

    losing_instances = []
    for player in players:
        instances = player.decision_function.instances
        if player.winner:
            # winning_actions.put(instances)
            winning_instances = instances
        else:
            losing_instances.extend(instances)
    
    # losing_actions.put(losing_instances)
    queue.put((id, player.winner, winning_instances, losing_instances))
    done_event.wait()

def play_round_async(
        curr_model: torch.nn.Module, list_of_models: List[torch.nn.Module], winning_actions: Queue, 
        losing_actions: Queue, device='cuda', eps=0.1):
    """
    Same as play_round, but puts everythigng onto the cpu and doesn't use events and done_queue
    """
    player = TrainingDecisionFunction(curr_model, device=device, eps=eps)
    adversaries = [TrainingDecisionFunction(model, device=device, eps=eps) for model in random.choices(list_of_models, k=3)]
    players = [player] + adversaries
    players = [Player(i, p) for i, p in zip(range(4), players)]

    game = Game(players)
    game.play()

    losing_instances = []
    for player in players:
        instances = [(t[0].to('cpu'), t[1].to('cpu')) for t in player.decision_function.instances]
        if player.winner:
            winning_actions.put(instances)
        else:
            losing_instances.extend(instances)
    losing_actions.put(losing_instances)
    return player.winner


def main(
    pool_size=2, batch_size=20, epochs=1000, er_mult=4, output_dir="./models", save_steps=500, method="process", model="a2c", 
    hidden_dim=256, lr_actor=1e-3, lr_critic=1e-3, alpha=1e-3, gamma=0.99):
    
    model_dispatch = {
        "base": Base,
        "ac": D2RLAC,
        "a2c": D2RLA2C,
        "aqc": D2RLAQC,
        "a2qc": D2RLA2QC
    }
    
    ModelClass = model_dispatch[model]
    model = ModelClass(hidden_dim=hidden_dim)
    
    train(model, epochs=epochs, batch_size=batch_size, 
          experience_replay_mult=er_mult, method=method, lr_actor=lr_actor, lr_critic=lr_critic,
          gamma=gamma, alpha=alpha, pool_size=pool_size, save_steps=save_steps, model_dir=output_dir)
    
    return model

if __name__ == "__main__":
    set_start_method("spawn")
    set_sharing_strategy("file_system")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    try:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, rlimit[1]), rlimit[1]))
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Train PUSOY model.")

    
    parser.add_argument("-p", "--pool_size", 
        help="Number of CPU processes to spawn. Defaults to 2.",
        type=int, default=2)
    parser.add_argument("-b", "--batch_size", 
        help="Batch size. Defaults to 20.",
        type=int, default=20)
    parser.add_argument("-e", "--epochs", 
        help="Training epochs. Defaults to 1000.",
        type=int, default=1000)
    parser.add_argument("--er_mult",
        help="Experience replay mult. Defaults to 4.",
        type=int, default=4)
    parser.add_argument("--output_dir", 
        help="Output directory. Defaults to ./models.",
        type=str, default="./models")
    parser.add_argument("--save_steps", 
        help="Steps to take before saving checkpoint. Defaults to 500",
        type=int, default=500)
    parser.add_argument("--method", 
        help="Whether to use process-based or pool-based implementation. Defaults to process.",
        type=str, default="process")
    
    parser.add_argument("-m", "--model",
        help="Model architecture to train. Defaults to A2C.",
        choices=["base", "ac", "a2c", "aqc", "a2qc"],
        type=str, default="a2c")
    parser.add_argument("--hidden_dim", 
        help="Change the hidden dim of the models. Defaults to 256.",
        type=int, default=256)
    parser.add_argument("--lr_actor", 
        help="Actor learning rate. Defaults to 1e-03 (0.001).",
        type=float, default=1e-03)
    parser.add_argument("--lr_critic", 
        help="Critic learning rate. Defaults to 1e-03 (0.001).",
        type=float, default=1e-03)
    parser.add_argument("--alpha", 
        help="Alpha, for entropy. Defaults to 1e-03 (0.001).",
        type=float, default=1e-03)
    parser.add_argument("--gamma", 
        help="Discount rate. Defaults to 0.99.",
        type=float, default=0.99)

    args = parser.parse_args()
    main(**vars(args))

