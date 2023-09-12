import torch
from pusoy.game import Game
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.models import D2RLA2C
import sys

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    model = D2RLA2C()
    model.load_state_dict(torch.load(checkpoint_path))

    game = Game.init_from_decision_functions([
        TrainingDecisionFunction(model),
        TrainingDecisionFunction(model),
        TrainingDecisionFunction(model),
        TrainingDecisionFunction(model) 
    ], debug=True)
    
    game.play()