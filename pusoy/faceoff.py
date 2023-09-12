import torch
from torch.multiprocessing import Pool, set_start_method
from models import D2RLA2C
from decision_function import TrainingDecisionFunction
from game import Game
from player import Player

def play_round_async(
        list_of_models: [torch.nn.Module], device='cuda'):
    """
    Same as play_round, but puts everything onto the cpu and doesn't use events and done_queue
    """
    with torch.no_grad():
        decision_functions = [TrainingDecisionFunction(model, device=device) for model in list_of_models]
        players = [Player(i, p) for i, p in enumerate(decision_functions)]
        game = Game(players)
        game.play()   
    return torch.tensor([player.winner for player in players], dtype=int)


def faceoff_checkpoints(checkpoint_paths):
    models = [D2RLA2C(), D2RLA2C(), D2RLA2C(), D2RLA2C()]
    for i in range(4):
        models[i].load_state_dict(torch.load(checkpoint_paths[i]))
    
    results = []
    pool = Pool(processes=3)
    for i in range(1000):
        pool.apply_async(play_round_async, args=[models], callback=(
            lambda result: results.append(result)
        ))
    pool.close()
    pool.join()
    results = torch.sum(torch.stack(results), dim=0) / len(results)
    print(checkpoint_paths)
    print(results)
    print(f"Winner: {checkpoint_paths[torch.argmax(results).item()]}")


if __name__ == "__main__":
    set_start_method("spawn")
    models = [6300, 8700, 15600, 19200, 20400]
    for i in range(0, len(models)-3, 4):
        checkpoints = [f"models/{models[i+u]}.pt" for u in range(4)]
        faceoff_checkpoints(checkpoints)
    checkpoints = [f"models/{models[-i]}.pt" for i in range(1, 5)]
    faceoff_checkpoints(checkpoints)
    
    