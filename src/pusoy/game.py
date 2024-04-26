from queue import Queue

import torch

from pusoy.player import Player
from pusoy.utils import Hands, RoundType


class Game:
    """Class that represents an entire game"""

    def __init__(self, players, debug=False):
        """Initializes a game with four players."""
        self.players = players
        self.player_queue = Queue(maxsize=4)
        for p in self.players:
            self.player_queue.put(p)
            p.game = self

        self.played_cards = [torch.zeros(52)] * 4

        cards = torch.randperm(52)
        for idx, p in enumerate(self.players):
            p.cards[cards[idx * 13 : (idx + 1) * 13]] = 1

        self.is_first_move = True
        self.finished = False
        self.winner = None

        self.prev_player = Player(None, None)
        self.prev_play = None
        self.curr_player = self.player_queue.get()
        while not self.curr_player.cards[0]:
            self.rotate_player()
        self.round_type = RoundType.NONE
        self.hand_type = Hands.NONE
        self.debug = debug

        if debug:
            print("New game created! To play this game, use game.play().")

    def play(self):
        while not self.finished:
            self.curr_player.play_round(self.debug, self.is_first_move)
            if self.is_first_move:
                self.is_first_move = False
            self.rotate_player()

    def rotate_player(self):
        self.player_queue.put(self.curr_player)
        self.curr_player = self.player_queue.get()
        while self.curr_player.passed:
            self.player_queue.put(self.curr_player)
            self.curr_player = self.player_queue.get()

        # If everyone passes, give control to new player
        if self.prev_player == self.curr_player:
            if self.debug:
                print(f"Control is now with {self.curr_player}")
            for p in self.players:
                p.passed = False
            self.prev_play = None
            self.round_type = RoundType.NONE
            self.hand_type = Hands.STRAIGHT

    def new_round(self):
        self.round_type = RoundType.NONE
        self.hand_type = Hands.STRAIGHT
        for p in self.players:
            p.passed = False

    def get_num_cards(self):
        num_cards = []
        for player in self.players:
            num_cards.append(len(player.cards))
        return num_cards

    def finish(self, player):
        self.finished = True
        player.winner = True
