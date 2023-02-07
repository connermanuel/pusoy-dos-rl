from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import torch

class Suit(Enum):
    clubs = 0
    spades = 1
    hearts = 2
    diamonds = 3

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.value < other.value
        return NotImplemented
    

class Value(Enum):
    three = 0
    four = 1
    five = 2
    six = 3
    seven = 4
    eight = 5
    nine = 6
    ten = 7
    jack = 8
    queen = 9
    king = 10
    ace = 11
    two = 12

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.value < other.value
        return NotImplemented

class Hands(Enum):
    NONE = 0
    STRAIGHT = 1
    FLUSH = 2
    FULL_HOUSE = 3
    FOUR_OF_A_KIND = 4
    STRAIGHT_FLUSH = 5

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def to_tensor(self, device):
        tensor = torch.zeros(6, dtype=torch.bool, device=device)
        tensor[self.value] = True
        return tensor[1:]

class Card():
    """Class that represents a single card."""
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
    
    def __str__(self):
        code = card_to_string(self)
        return f'[{code}] The {self.value.name} of {self.suit.name}'
    
    def __lt__(self, other):
        if self.__class__ == other.__class__:
            if self.value == other.value:
                return self.suit < other.suit
            return self.value < other.value
        return NotImplemented
    
    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.value == other.value and self.suit == other.suit
        return NotImplemented
    
    def __hash__(self):
        return int((self.value.value * 4) + self.suit.value)

def idx_to_card(idx):
    value = idx // 4
    suit = idx % 4
    return Card(Value(value), Suit(suit))

class RoundType(Enum):
    NONE = 0
    SINGLES = 1
    PAIRS = 2
    TRIPLES = 3
    FOURS = 4 # for compatibility only
    HANDS = 5

    def __str__(self):
        return self.name
    
    def to_tensor(self, device):
        tensor = torch.zeros(5, dtype=torch.bool, device=device)
        value = min(self.value, 4)
        tensor[value] = True
        return tensor


def string_to_card(card_str):
    value_str = card_str[:-1]
    value_number = int(faces_to_nums.get(value_str, value_str))
    value = Value((value_number - 3) % 13)

    suit_str = card_str[-1]
    suit = Suit(suit_dict[suit_str])

    return Card(value, suit)

def card_to_string(card):
    value = (card.value.value + 3) % 13
    value = nums_to_faces.get(value, value)

    suit = card.suit.name[0].upper()

    return str(value) + suit

def print_cards(card_list):
    print('')
    idxs = torch.nonzero(card_list).flatten()
    for idx in idxs:
        print(idx_to_card(idx.item()))
    print('')

faces_to_nums = {'J': 11, 'Q': 12, 'K': 13, 'A': 14}
suit_dict = {'C': Suit.clubs, 'S': Suit.spades, 'H': Suit.hearts, 'D': Suit.diamonds}

nums_to_faces = {11: 'J', 12: 'Q', 0: 'K', 1: 'A'}

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

def count_cards_per_value(cards):
    return cards.reshape(13, 4).sum(dim=1)

def card_exists_per_value(cards):
    return cards.reshape(13, 4).max(dim=1)[0]

def count_cards_per_suit(cards):
    return cards.reshape(13, 4).sum(dim=0)
