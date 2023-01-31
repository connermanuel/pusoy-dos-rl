from pusoy.game import Game, DummyGame
from pusoy.utils import Card, Suit, Value

def test_diff_cards():
    game = DummyGame()
    players = game.players
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            assert set(players[i].cards()).isdisjoint(set(players[j].cards))

def test_all_cards_distributed():
    game = DummyGame()
    all_sets = set()
    for player in game.players:
        all_sets = set.union(all_sets, player.cards)
    cards = set([Card(suit, value) for suit in Suit for value in Value])
    assert cards == all_sets

        
