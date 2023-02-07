"""
Uses the whale optimization algorithm to find the best hyperparameters for a specified ModelClass.
Hyperparameters:
 -- Hidden Size: 2^n (default is n=6, or 256)
 -- Actor Learning rate: 10^n (default is -3.5, or 3e-3)
 -- Critic Learning rate: 10^n (default is -3, or 1e-3)
 -- Alpha (entropy): 10^n (default is -3, or 1e-3)
 -- Discount Factor: (1 - 10^n)) (default is -2, or .99) (note: bounded above by -0.5)
"""
from pusoy.models import Base, D2RLAC, D2RLA2C, D2RLA2QC, D2RLA2QC
from pusoy.train import main as tr_main
from pusoy.player import Player
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.game import Game

import torch
from torch.multiprocessing import Pool

import math
from typing import Type

POOL_SIZE = 2
BATCH_SIZE = 20
EPOCHS = 500
ER_MULT = .4
SAVE_STEPS = 500

NUM_PARAMS = 5

def build_model_from_args(model_class: str, args: torch.Tensor, output_dir: str):
    """Builds and trains a model of the specified type corresponding to the args, and saves to directory."""

    hidden_dim = int(2**args[0])
    lr_actor = 10**args[1]
    lr_critic = 10**args[2]
    alpha = 10**args[3]
    gamma = 1 - 10**args[4]

    model = tr_main(pool_size=2, batch_size=20, epochs=1000, er_mult=4, output_dir=output_dir, save_steps=500, model=model_class, 
    hidden_dim=hidden_dim, lr_actor=lr_actor, lr_critic=lr_critic, alpha=alpha, gamma=gamma)

    return model
    
def faceoff_single(model_0, model_1, model_2, model_3):
    """Takes four models and makes them play a round against each other. Returns a one-hot array of the winner."""
    models = [model_0, model_1, model_2, model_3]
    players = [Player(i, TrainingDecisionFunction(models[i]))  for i in range(4)]
    game = Game(players)
    game.play()
    return torch.tensor([player.winner for player in players], dtype=int)

def faceoff(model_0, model_1, model_2, model_3, n_rounds=1000):
    """Takes four models and makes them play n_rounds against each other. Returns an array of win percentages."""
    models = [model_0, model_1, model_2, model_3]

    with Pool(POOL_SIZE) as pool:
        games = [pool.apply_async(faceoff_single, args=models) for i in range(n_rounds)]
        results = [result.get(timeout=60) for result in games]
        return torch.sum(results) / n_rounds

def whales(n_whales: int, n_iters: int):
    """
    Performs Whale Optimization Algorithm for n_iters to find the best model.
    Each whale is a tensor of hyperparameters with which a model can be built and instantiated.
    """
    ## init whales
    whales = []
    ## faceoff whales
    ## best_whale = best whale
    for iter in range(n_iters):
        for idx, whale in enumerate(whales):
            ## update a, A, C l, and p
            if torch.rand(1).item() < 0.5:
                A, C = compute_A_and_C(a)
                if torch.linalg.norm(A) < 1:
                    target = best_whale
                else:
                    target = (whales[:idx] + whales[idx+1:])[torch.randint(n_whales-1, (1,)).item()]
                encircle(whale, target, A, C)
            else:
                spiral(whale, best_whale)
            whale = fix(whale)
        faceoff whales
        best_whale = best whale
    
    return best_whale

def compute_A_and_C(a):
    r = torch.rand(NUM_PARAMS)
    A = (2.0 * a * r) - a
    C = 2 * r
    return A, C

# Note: encircle is the same as explore, just with a different choice of whale
def encircle(whale, target, A, C):
    D = torch.linalg.norm((C * target) - whale)
    return target * (A-D)

def spiral(whale, best_whale):
    D = torch.linalg.norm(best_whale - whale)
    L = (torch.rand(NUM_PARAMS) * 2) - 1
    return ((D * torch.exp(0.5*L)) * torch.cos(2.0*math.pi*L)) + best_whale