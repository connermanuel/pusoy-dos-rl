from pusoy.game import Game
from pusoy.player import Player
from pusoy.models import D2RLModelWithCritic, D2RLModelWithQValueCritic, DumbModel, BaseModel, D2RLModel
from pusoy.decision_function import Neural, TrainingDecisionFunction
from torch import load
import sys

class ModelVSDummies(Game):
    def __init__(self, path1):

        model = D2RLModelWithCritic()
        model.load_state_dict(load(path1))
        you = Player(0, Neural(model, eps=0.1, debug=True))

        adv_model = D2RLModelWithCritic()
        adversaries = [Player(i, Neural(adv_model)) for i in range(2, 4)]
        players = [you]  + adversaries
        super().__init__(players, debug=True)

if __name__ == '__main__':
    game = ModelVSDummies(sys.argv[1], sys.argv[2])
    game.play()