import torch
from pusoy.game import Game
from pusoy.decision_function import Interactive, Neural
from pusoy.models import D2RLA2C
import sys

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    model = D2RLA2C()
    model.load_state_dict(torch.load(checkpoint_path))

    game = Game.init_from_decision_functions([
        Interactive(),
        Neural(model),
        Neural(model),
        Neural(model) 
    ], debug=True)
    
    game.play()