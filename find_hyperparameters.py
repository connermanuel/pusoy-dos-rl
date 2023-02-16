"""
Uses the whale optimization algorithm to find the best hyperparameters for a specified ModelClass.
Hyperparameters:
 -- Hidden Size: 2^n (default is n=8, or 256)
 -- Actor Learning rate: 10^n (default is -3.5, or 3e-3)
 -- Critic Learning rate: 10^n (default is -3, or 1e-3)
 -- Alpha (entropy): 10^n (default is -3, or 1e-3)
 -- Discount Factor: (1 - 10^n)) (default is -2, or .99) (note: bounded above by -0.5)
"""
from pusoy.train import main as tr_main
from pusoy.player import Player
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.game import Game
from pusoy.models import Base, D2RLAC, D2RLA2C, D2RLAQC, D2RLA2QC

import torch
from torch.multiprocessing import Pool, set_start_method, set_sharing_strategy

import os
import joblib
import argparse
import math
import random

NUM_PARAMS = 5

model_dispatch = {
    "base": Base,
    "ac": D2RLAC,
    "a2c": D2RLA2C,
    "aqc": D2RLAQC,
    "a2qc": D2RLA2QC
}

def build_model_from_args(model_class: str, whale_args: torch.Tensor, args: argparse.Namespace):
    """Builds and trains a model of the specified type corresponding to the args, and saves to directory."""

    hidden_dim = int(2**whale_args[0])
    lr_actor = 10**whale_args[1]
    lr_critic = 10**whale_args[2]
    alpha = 10**whale_args[3]
    gamma = 1 - 10**whale_args[4]

    output_dir = f"{args.output_dir}/{str(whale_args)[7:-1]}"

    if os.path.exists(f"{output_dir}/{args.epochs}.pt"):
        print(f"Loading whale from checkpoint: {whale_args}")
        model = model_dispatch[model_class](hidden_dim = hidden_dim)
        model.load_state_dict(torch.load(f"{output_dir}/{args.epochs}.pt"))
    else:
        print(f"Training whale: {whale_args}")
        model = tr_main(pool_size=args.pool_size, batch_size=args.batch_size, epochs=args.epochs, er_mult=args.er_mult, save_steps=args.save_steps,
        method=args.method, output_dir=output_dir, model=model_class, hidden_dim=hidden_dim, lr_actor=lr_actor, lr_critic=lr_critic, alpha=alpha, gamma=gamma)

    return model
    
def faceoff_single(model_0, model_1, model_2, model_3):
    """Takes four models and makes them play a round against each other. Returns a one-hot array of the winner."""
    models = [model_0, model_1, model_2, model_3]
    players = [Player(i, TrainingDecisionFunction(models[i]))  for i in range(4)]
    game = Game(players)
    game.play()
    return torch.tensor([player.winner for player in players], dtype=int)

def faceoff_four(models: list, args=argparse.Namespace):
    """Takes four models and makes them play n_rounds against each other. Returns an array of win percentages."""
    n_rounds = args.epochs
    with Pool(args.pool_size) as pool:
        results = [pool.apply(faceoff_single, args=models) for i in range(n_rounds)]
        return torch.stack(results).sum(dim=0) / n_rounds

def faceoff(models: list, args=argparse.Namespace):
    """Takes at least 4 models and makes them play n_rounds against each other. Returns win percentages from the final round"""
    print("First faceoff")
    first_four_results = faceoff_four(models[:4], args)
    print("Second faceoff")
    next_four_results = faceoff_four(models[-4:], args)
    top_idxs = torch.cat([
        torch.argsort(first_four_results, descending=True)[:2],
        torch.argsort(next_four_results, descending=True)[:2] + (len(models)-4)]).long()
    print("Final faceoff")
    final_four_results = faceoff_four([models[i] for i in top_idxs], args)
    wins = torch.zeros(len(models))
    wins[top_idxs] = final_four_results
    return wins

def compute_A_and_C(a):
    r = torch.rand(NUM_PARAMS)
    A = (2.0 * a * r) - a
    C = 2 * r
    return A, C

# Note: encircle is the same as explore, just with a different choice of whale
def encircle(whale, target, A, C):
    D = torch.linalg.norm((C * target) - whale)
    return target - (A*D)

def spiral(whale, best_whale):
    D = torch.linalg.norm(best_whale - whale)
    L = (torch.rand(NUM_PARAMS) * 2) - 1
    return ((D * torch.exp(0.5*L)) * torch.cos(2.0*math.pi*L)) + best_whale

def fix_whale(whale):
    maxes = torch.tensor([10, -1, -1, -1, -0.5])
    mins = torch.tensor([6, -10, -10, -10, -10])
    whale = torch.where(whale < maxes, whale, maxes)
    whale = torch.where(whale > mins, whale, mins)
    return whale

def whales(args: argparse.Namespace):
    """
    Performs Whale Optimization Algorithm for n_iters to find the best model.
    Each whale is a tensor of hyperparameters with which a model can be built and instantiated.
    """
    n_whales, n_iters, model_class = args.n_whales, args.n_iters, args.model

    default = torch.tensor([8, -3.5, -3, -3, -2])
    whales = [default]
    if os.path.exists(args.output_dir):
        other_whales = [torch.tensor(eval(whale_str)) for whale_str in os.listdir(args.output_dir) if whale_str != str(default)[7:-1]]
        random.shuffle(other_whales)
        whales = whales + other_whales[:3]
    
    for _ in range(n_whales - len(whales)):
        whales.append(torch.tensor([8, -3.5, -3, -3, -2]) + (torch.rand(NUM_PARAMS) - 0.5))

    models = [build_model_from_args(model_class=model_class, whale_args=whale, args=args) for whale in whales]
    results = faceoff(models, args)
    best_whale_idx = torch.argmax(results)
    best_whale = whales[best_whale_idx]
    best_whales = [best_whale]
    a = 1
    step_size = a / n_iters

    for iter in range(n_iters):
        print(f"Iter: {iter}")
        print(f"The best whale is {best_whale}")
        for idx, whale in enumerate(whales):
            if idx == best_whale_idx:
                continue
            if torch.rand(1).item() < 0.5:
                A, C = compute_A_and_C(a)
                if torch.linalg.norm(A) < 0.4:
                    print("encircle")
                    target = best_whale
                else:
                    print("explore")
                    target = (whales[:idx] + whales[idx+1:])[torch.randint(n_whales-1, (1,)).item()]
                whale = encircle(whale, target, A, C)
            else:
                print("spiral")
                whale = spiral(whale, best_whale)
            whale = fix_whale(whale)
            whales[idx] = whale
            old_model = models[idx]
            models[idx] = build_model_from_args(model_class=model_class, whale_args=whale, args=args)
            del old_model
        print("Commencing faceoff")
        results = faceoff(models, args)
        best_whale_idx = torch.argmax(results)
        best_whale = whales[best_whale_idx]
        best_whales.append(best_whale)
        a -= step_size
    
    return best_whales

def main(args):
    if args.pool_size == 0:
        args.pool_size = torch.multiprocessing.cpu_count()
    args.output_dir = args.output_dir + f"/{args.model}"
    best_whales = whales(args)
    joblib.dump(best_whales, f"{args.output_dir}/best_whales.pkl")

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
        help="Number of CPU processes to spawn. Defaults to number of cpu cores.",
        type=int, default=0)
    parser.add_argument("-b", "--batch_size", 
        help="Batch size. Defaults to 20.",
        type=int, default=20)
    parser.add_argument("-e", "--epochs", 
        help="Training epochs. Defaults to 250.",
        type=int, default=250)
    parser.add_argument("--er_mult",
        help="Experience replay mult. Defaults to 4.",
        type=int, default=4)
    parser.add_argument("--output_dir", 
        help="Output directory. Defaults to ./models.",
        type=str, default="./models")
    parser.add_argument("--save_steps", 
        help="Steps to take before saving checkpoint. Defaults to 250",
        type=int, default=250)
    parser.add_argument("--method", 
        help="Whether to use process-based or pool-based implementation. Defaults to process.",
        choices=["process", "pool"],
        type=str, default="process")
    
    parser.add_argument("-m", "--model",
        help="Model architecture to train. Defaults to A2C.",
        choices=["base", "ac", "a2c", "aqc", "a2qc"],
        type=str, default="a2c")
    parser.add_argument("--n_whales", 
        help="Number of whales. Defaults to 4",
        type=int, default=4)
    parser.add_argument("--n_iters", 
        help="Number of iterations. Defaults to 8",
        type=int, default=8)

    args = parser.parse_args()
    main(args)