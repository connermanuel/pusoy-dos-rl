from pusoy.utils import Card
from pusoy.decision_function import Interactive
import torch

class Player():
    def __init__(self, number, decision_function):
        self.cards = torch.zeros(52)
        self.number = number
        self.passed = False
        self.decision_function = decision_function
        self.winner = False
    
    def give_cards(self, cards):
        """
        Takes in a list of CARD objects and adds them to the player's CARDS vector."""
        for card in cards:
            self.cards[card.__hash__()] = 1

    def play_round(self, game, debug=False, is_first_move=False):
        while True:
            try:
                action = self.decision_function.play(
                    self.number,
                    self.cards,
                    game.round_type,
                    game.hand_type,
                    game.prev_play,
                    game.prev_player.number,
                    game.played_cards,
                    is_first_move
                )
                return action
            except ValueError as ve:
                print(ve)
    
    def __str__(self):
        return f"player {self.number}"

class InteractivePlayer(Player):
    def __init__(self, number, decision_function):
        super().__init__(number, Interactive())

