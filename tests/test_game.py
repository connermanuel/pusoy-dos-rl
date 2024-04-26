from pusoy.decision_function import Neural
from pusoy.game import Game
from pusoy.player import Player


def test_game_neural_basic(base_model_a2c):
    for _ in range(100):
        decision_functions = [Neural(base_model_a2c) for i in range(4)]
        players = [Player(i, func) for i, func in enumerate(decision_functions)]
        game = Game(players, debug=True)
        game.play()
        pass
