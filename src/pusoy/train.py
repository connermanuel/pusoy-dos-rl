from pusoy.game import Game
from pusoy.player import Player
from pusoy.models import PusoyModel, FullLSTMModel
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.losses import ppo_loss
from pusoy.utils import ExperienceBuffer

import torch
from torch.multiprocessing import (
    Pool,
    cpu_count,
    set_start_method,
    set_sharing_strategy,
)
from torch.distributions.binomial import Binomial
import torch.nn.functional as F

import os
import argparse
import copy
from collections import Counter
import pickle as pkl

from typing import List, Type

DISTRIBUTION = Binomial(3, 0.2)
ETA = 0.01

def train(
    ModelClass: Type[PusoyModel],
    hidden_size: int = 256,
    num_models: int = 8,
    epochs: int = 1500,
    batch_size: int = 20,
    lr_actor: float = 0.001,
    lr_critic: float = 0.001,
    eps: float = 0.1,
    opponent_steps: int = 10,
    gamma: float = 0.99,
    lambd: float = 0.9,
    c_entropy: float = 0.001,
    beta: float = 0.1,
    pool_size: int = 4,
    save_steps: int = 500,
    checkpoint: str = None,
    device: str = "cuda",
    model_dir = None,
):
    """
    Trains curr_model using self-play.

    Parameters:
    ModelClass -- class constructor for models
    hidden_size -- hidden dimension of models used
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
    c_entropy -- entropy factor
    beta -- exploration parameter
    pool_size -- number of processes to spawn
    save_steps -- number of steps to take before saving
    device -- device to perform operations on
    model_dir -- where to save the models
    """
    start = 0
    torch.autograd.set_detect_anomaly(True)

    train_model = ModelClass(hidden_size=hidden_size)
    train_model.to(device)
    opt = torch.optim.Adam(train_model.parameters(), lr_actor)
    past_models = {
        "models": [create_copy(ModelClass, hidden_size, train_model)],
        "qualities": torch.tensor([0.0]),
    }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if checkpoint:
        train_model, opt, past_models, start = load_from_checkpoint(train_model, opt, model_dir, checkpoint)
    
    prev_model = create_copy(ModelClass, hidden_size, train_model)
    rollout_model = create_copy(ModelClass, hidden_size, train_model)

    buffer = ExperienceBuffer()

    print(f"Beginning training")
    print(f"Cpu count: {cpu_count()}")

    res = [None] * batch_size
    pool = Pool(processes=pool_size)

    for epoch in range(start, epochs):
        epoch_buffer = ExperienceBuffer()
        for i in range(batch_size):
            models, past_model_idxs = select_models(rollout_model, past_models)
            args = [models, past_model_idxs, device, eps]
            res[i] = pool.apply_async(
                play_round_async,
                args=args,
                callback=(lambda result: pool_callback(result, epoch_buffer)),
            )
            # result = play_round_async(*args)
            # pool_callback(result, epoch_buffer)

        if not buffer.is_empty():
            train_step(
                train_model,
                prev_model,
                opt,
                buffer,
                device,
                eps,
                gamma,
                lambd,
                c_entropy,
            )
        
        prev_model.load_state_dict(rollout_model.state_dict())
        past_models = update_qualities(epoch_buffer, past_models)
        if not (epoch + 1) % opponent_steps:
            past_models = update_opponents(
                past_models, ModelClass, train_model, hidden_size, num_models
            )
        
        [r.wait(timeout=10) for r in res]
        buffer = epoch_buffer
        rollout_model.load_state_dict(train_model.state_dict())

        print(f"Epoch {epoch} winrate: {buffer.wins/batch_size}")
        if not (epoch + 1) % save_steps:
            save_checkpoint(train_model, opt, model_dir, past_models, epoch + 1)

def load_from_checkpoint(train_model, opt, model_dir, checkpoint):
    print("Loading from checkpoint.")
    train_model.load_state_dict(torch.load(f"{model_dir}/{checkpoint}/model.pt"))
    opt.load_state_dict(torch.load(f"{model_dir}/{checkpoint}/optim.pt"))
    with open(f"{model_dir}/{checkpoint}/opponents.pkl", "rb") as f:
        past_models = pkl.load(f)
    start = int(checkpoint)
    print("Successfully loaded from checkpoint.")
    return train_model, opt, past_models, start

def save_checkpoint(train_model, opt, model_dir, past_models, epoch):
    print("Saving...")
    os.mkdir(f"{model_dir}/{epoch}")
    torch.save(train_model.state_dict(), f"{model_dir}/{epoch}/model.pt")
    torch.save(opt.state_dict(), f"{model_dir}/{epoch}/optim.pt")
    with open(f"{model_dir}/{epoch}/opponents.pkl", "wb") as f:
        pkl.dump(past_models, f)
    print("Save complete.")

def select_models(curr_model, past_models):
    """
    Given previous models and their qualities, returns a list of models that will participate in the 
    next round of pusoy dos.    
    """
    n_past_models = int(DISTRIBUTION.sample())
    models = [curr_model for _ in range(4 - n_past_models)]
    past_model_idxs = []
    if n_past_models:
        past_model_idxs = torch.multinomial(
            torch.exp(past_models["qualities"]), n_past_models, replacement=True
        ).tolist()
        models.extend([past_models["models"][idx] for idx in past_model_idxs])
    return models, past_model_idxs

def train_step(
    train_model: torch.nn.Module, 
    prev_model: torch.nn.Module, 
    opt: torch.optim.Optimizer,
    buffer: ExperienceBuffer, 
    device: str, 
    eps: float=0.1, 
    gamma: float=0.99,
    lambd: float=0.9, 
    c_critic: float=1,
    c_entropy: float=0.001
):
    
    win_rewards = [torch.zeros(len(l), device=device) for l in buffer.win_inputs]
    for t in win_rewards:
        t[-1] = 1
    lose_rewards = [torch.zeros(len(l), device=device) for l in buffer.lose_inputs]
    for t in lose_rewards:
        t[-1] = -(len(win_rewards) / len(lose_rewards))

    inputs = buffer.win_inputs + buffer.lose_inputs
    actions = buffer.win_actions + buffer.lose_actions
    rewards = win_rewards + lose_rewards

    loss = ppo_loss(
        train_model,
        prev_model,
        inputs,
        actions,
        rewards,
        eps_clip=eps,
        gamma=gamma,
        lambd=lambd,
        c_critic=c_critic,
        c_entropy=c_entropy,
        device=device,
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

def play_round_async(
    models: list[torch.nn.Module],
    past_model_idxs: list[int],
    device="cuda",
    beta=0.1,
):
    """
    Plays a round of pusoy dos with predetermined opponents.
    Returns information about inputs and actions taken in order to train the model later.
    """
    decision_functions = [
        TrainingDecisionFunction(model, device, beta) for model in models
    ]
    players = [Player(i, func) for i, func in zip(torch.randperm(4), decision_functions)]
    game = Game(players)
    game.play()

    losing_instances = []
    for p in players:
        if p.winner:
            winning_instances = p.decision_function.instances
        else:
            losing_instances.append(p.decision_function.instances)
    return {
        "is_winner": players[0].winner,
        "winning_instances": winning_instances,
        "losing_instances": losing_instances,
        "past_model_idxs": past_model_idxs
    }


def update_qualities(buffer, past_models):
    """
    Update the qualities of the past models, based on how many times each model lost previously.
    """
    probabilities = F.softmax(past_models["qualities"], dim=0)
    for idx in buffer.loss_counter.keys():
        past_models["qualities"][idx] -= (
            buffer.loss_counter[idx] * ETA / (probabilities[idx] * len(probabilities))
        )
    return past_models


def update_opponents(past_models, ModelClass, curr_model, hidden_size, num_models):
    past_models["models"].append(create_copy(ModelClass, hidden_size, curr_model))
    past_models["qualities"] = torch.cat(
        (
            past_models["qualities"],
            torch.max(past_models["qualities"]).reshape(1),
        )
    )
    while len(past_models["models"]) > num_models:
        min_idx = torch.argmin(past_models["qualities"])
        past_models["models"] = (
            past_models["models"][:min_idx] + past_models["models"][min_idx + 1:]
        )
        past_models["qualities"] = torch.cat((
            past_models["qualities"][:min_idx], past_models["qualities"][min_idx + 1:]
        ))
    return past_models


def pool_callback(
    result: dict,
    buffer: ExperienceBuffer,
):
    buffer.update_wins(result["is_winner"])
    buffer.update_win_inputs(result["winning_instances"]["inputs"])
    buffer.update_win_actions(result["winning_instances"]["actions"])
    for instance in result["losing_instances"]:
        buffer.update_lose_inputs(instance["inputs"])
        buffer.update_lose_actions(instance["actions"])
    if result["is_winner"]:
        buffer.update_lose_counter(result["past_model_idxs"])


def create_copy(
    ModelClass: [torch.nn.Module],
    hidden_size: int,
    model: torch.nn.Module,
    requires_grad: bool = False,
):
    model_copy = ModelClass(hidden_size=hidden_size)
    model_copy.requires_grad_(requires_grad)
    model_copy.load_state_dict(model.state_dict())
    return model_copy


def create_index(
    experience_replay_size: int,
    batch_size: int,
    batches_available: int,
    target_batches: int,
):
    batch_number = -torch.log2(torch.rand(experience_replay_size))
    batch_number = torch.trunc(
        torch.clamp(batch_number, max=min(batches_available, target_batches))
    )
    unit_number = torch.randint(low=0, high=batch_size, size=(experience_replay_size,))
    idxs = (batch_number * batch_size) + unit_number
    idxs = idxs.to(torch.int)
    return -idxs + 1


def main(
    pool_size,
    batch_size,
    epochs,
    num_models,
    output_dir,
    save_steps,
    checkpoint,
    opponent_steps,
    model,
    hidden_size,
    lr_actor,
    lr_critic,
    c_entropy,
    beta,
    eps,
    gamma,
    lambd,
):
    model_dispatch = {
        "lstm": FullLSTMModel,
    }

    ModelClass = model_dispatch[model]

    train(
        ModelClass,
        hidden_size,
        num_models=num_models,
        epochs=epochs,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        lambd=lambd,
        c_entropy=c_entropy,
        beta=beta,
        eps=eps,
        pool_size=pool_size,
        save_steps=save_steps,
        checkpoint=checkpoint,
        opponent_steps=opponent_steps,
        model_dir=output_dir,
    )

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

    parser.add_argument(
        "-p",
        "--pool_size",
        help="Number of CPU processes to spawn. Defaults to 4.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size. Defaults to 64.", type=int, default=64
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Training epochs. Defaults to 100000.",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--num_models",
        help="Number of models to keep. Defaults to 8.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory. Defaults to ./models.",
        type=str,
        default="./models",
    )
    parser.add_argument(
        "--save_steps",
        help="Steps to take before saving checkpoint. Defaults to 256",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--opponent_steps",
        help="Steps to take before storing an opponent as an adversary. Defaults to 32",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to load from.",
        type=str,
        default=None
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model architecture to train. Defaults to LSTM.",
        choices=["lstm"],
        type=str,
        default="lstm",
    )
    parser.add_argument(
        "--hidden_size",
        help="Change the hidden size of the models. Defaults to 256.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--lr_actor",
        help="Actor learning rate. Defaults to 1e-04 (0.0001).",
        type=float,
        default=1e-04,
    )
    parser.add_argument(
        "--lr_critic",
        help="Critic learning rate. Defaults to 1e-04 (0.0001).",
        type=float,
        default=1e-04,
    )
    parser.add_argument(
        "--c_entropy",
        help="Entropy coefficient. Defaults to 1e-02 (0.01).",
        type=float,
        default=1e-02,
    )
    parser.add_argument(
        "--beta",
        help="Beta, for exploration. Defaults to 0.01.",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--eps", help="Epsilon, for clipping. Defaults to 0.1.", type=float, default=0.1
    )
    parser.add_argument(
        "--gamma",
        help="Discount rate for computing loss. Defaults to 0.99.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--lambd",
        help="Hyperparameter for computing GAE. Defaults to 0.92.",
        type=float,
        default=0.92,
    )

    args = parser.parse_args()
    main(**vars(args))
