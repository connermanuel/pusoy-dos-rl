import torch
from torch.multiprocessing import Pool, set_start_method
from pusoy.models import A2CLSTM
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.game import Game
from pusoy.player import Player

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
    models = [A2CLSTM(), A2CLSTM(), A2CLSTM(), A2CLSTM()]
    for i in range(4):
        if checkpoint_paths[i]:
            models[i].load_state_dict(torch.load(checkpoint_paths[i]))
    
    results = []
    pool = Pool(processes=4)
    play_round_async(models)
    for i in range(2000):
        pool.apply_async(play_round_async, args=[models], callback=(
            lambda result: results.append(result)
        ))
    pool.close()
    pool.join()
    print(len(results))
    results = torch.sum(torch.stack(results), dim=0) / len(results)
    print(checkpoint_paths)
    print(results)
    print(f"Winner: {checkpoint_paths[torch.argmax(results).item()]}")


if __name__ == "__main__":
    set_start_method("spawn")
    models = ["2560", "", "", ""]
    checkpoints = [f"models/{model}/model.pt" if model else None for model in models ]
    faceoff_checkpoints(checkpoints)
    
    