from pusoy.game import Game
from pusoy.player import Player
from pusoy.decision_function import Interactive

class DecisionFunctionGame(Game):
    def __init__(self, decision_function, debug=True):
        players = [Player(i, decision_function) for i in range(4)]
        super().__init__(players, debug=debug)

class InteractiveGame(DecisionFunctionGame):
    def __init__(self):
        super().__init__(Interactive())

if __name__ == "__main__":
    game = InteractiveGame()
    game.play()