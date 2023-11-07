import torch
from pusoy.game import Game
from pusoy.decision_function import Interactive, Neural
from pusoy.models import A2CLSTM
import sys

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    model = A2CLSTM()
    model.load_state_dict(torch.load(checkpoint_path))

    game = Game.init_from_decision_functions([
        Interactive(),
        Neural(model),
        Neural(model),
        Neural(model) 
    ], debug=True)
    
    game.play()