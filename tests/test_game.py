from pusoy.constants import DEVICE
from pusoy.decision_function import TrainingDecisionFunction
from pusoy.game import Game
from pusoy.models import DenseA2C


def test_game():
    model = DenseA2C()
    decision_functions = [
        TrainingDecisionFunction(model, DEVICE, beta) for model in models
    ]
    players = [Player(i, func) for i, func in enumerate(decision_functions)]
    game = Game(players)
    game.play()
