import torch
from pusoy.game import Game
from pusoy.decision_function import Neural, Interactive
from pusoy.models import FullLSTMModel
import sys
import os

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    while True:
        model = FullLSTMModel()

        game = Game.init_from_decision_functions([
            Neural(model),
            Neural(model),
            Neural(model),
            Neural(model) 
        ], debug=True)
        
        game.play()