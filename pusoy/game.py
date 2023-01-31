from pusoy.player import Player
from pusoy.utils import RoundType, Hands, print_cards
from pusoy.decision_function import Interactive, Neural, TrainingDecisionFunction
from pusoy.models import DumbModel, D2RLModelWithCritic

from queue import Queue

import numpy as np
import torch

class Game():
    """Class that represents an entire game"""

    def __init__(self, players, debug=False):
        """Initializes a game with four players."""
        #TODO: add options for player numbs
        self.players = players
        self.player_queue = Queue(maxsize=4)
        for p in self.players:
            self.player_queue.put(p)
            p.game = self
        
        self.played_cards = [torch.zeros(52)] * 4

        cards = np.arange(52)
        np.random.shuffle(cards)
        for idx, p in enumerate(self.players):
            p.cards[cards[idx*13: (idx+1)*13]] = 1
        if debug:
            for p in self.players:
                print(f"Player {p.number} starts with:")
                print_cards(p.cards)
        
        self.is_first_move = True
        self.finished = False
        self.winner = None

        self.prev_player = Player(None, None, None)
        self.prev_play = None
        self.curr_player = self.player_queue.get()
        while not self.curr_player.cards[0]:
            self.rotate_player()
        self.round_type = RoundType.NONE
        self.hand_type = Hands.STRAIGHT
        self.debug = debug

        if debug:
            print('New game created! To play this game, use game.play().')
    
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
                print(f'Control is now with {self.curr_player}')
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
    
    def init_from_decision_functions(decision_functions):
        players = [Player(None, i, decision_function) for i, decision_function in zip(range(4), decision_functions)]
        return Game(players)

class DecisionFunctionGame(Game):
    def __init__(self, decision_function, debug=False):
        players = [Player(self, i, decision_function) for i in range(4)]
        super().__init__(players, debug=debug)

class DummyGame(DecisionFunctionGame):
    def __init__(self):
        super().__init__(None)

class InteractiveGame(DecisionFunctionGame):
    def __init__(self):
        super().__init__(Interactive())

class DummyGame(DecisionFunctionGame):
    def __init__(self):
        super().__init__(Neural(DumbModel()), debug=True)
    
class DummyInteractiveNeuralGame(Game):
    def __init__(self):
        you = Player(self, 0, Interactive())
        adversaries = [Player(self, i, Neural(DumbModel())) for i in range(1, 4)]
        players = [you] + adversaries
        super().__init__(players)

class DummyInteractiveNeuralTrainingGame(Game):
    def __init__(self, path):
        you = Player(self, 0, Interactive())
        if path:
            adversaries = [Player(self, i, TrainingDecisionFunction(DumbModel())) for i in range(1, 4)]
        else:
            model = D2RLModelWithCritic()
            model.load_state_dict(torch.load(path))
            adversaries = [Player(self, i, Neural(model)) for i in range(1, 4)]
        players = [you] + adversaries
        super().__init__(players)