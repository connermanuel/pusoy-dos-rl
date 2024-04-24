import torch

from pusoy.decision_function import Interactive
from pusoy.utils import Card


class Player:
    """Participates in a Game of Pusoy."""

    def __init__(self, number, decision_function):
        """
        A player must be constructed with a
        """
        self.number = number
        self.cards = torch.zeros(52)
        self.passed = False
        self.decision_function = decision_function
        self.winner = False

    def __str__(self):
        return f"player {self.number}"
    
    def play()

    def play_round(self, debug=False, is_first_move=False):
        while True:
            try:
                action = self.decision_function.play(
                    self.number,
                    self.cards,
                    self.game.round_type,
                    self.game.hand_type,
                    self.game.prev_play,
                    self.game.prev_player.number,
                    self.game.played_cards,
                    is_first_move,
                )
                action.play(self, debug, is_first_move)
                return
            except ValueError as ve:
                print(ve)


class InteractivePlayer(Player):
    def __init__(self, game, number, decision_function):
        super().__init__(game, number, Interactive())
