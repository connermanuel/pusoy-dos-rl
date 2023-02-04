from pusoy.utils import Card
from pusoy.decision_function import Interactive
import torch

class Player():
    def __init__(self, game, number, decision_function):
        self.cards = torch.zeros(52)
        self.game = game
        self.number = number
        self.passed = False
        self.decision_function = decision_function
        self.winner = False
    
    def give_cards(self, cards):
        """
        Takes in a list of CARD objects and adds them to the player's CARDS vector."""
        for card in cards:
            self.cards[card.__hash__()] = 1
    
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
                    is_first_move
                )
                action.play(self, debug, is_first_move)
                return
            except ValueError as ve:
                print(ve)
    
    def __str__(self):
        return f"player {self.number}"

class InteractivePlayer(Player):
    def __init__(self, game, number, decision_function):
        super().__init__(game, number, Interactive())

