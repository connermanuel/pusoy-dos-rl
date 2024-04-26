from abc import ABC, abstractmethod

import torch

from pusoy.utils import (
    Hands,
    RoundType,
    count_cards_per_value,
    idx_to_card,
    print_cards,
)


class Action(ABC):
    """
    Parameters:
    cards: (52,) binary vector of cards as part of the action.
    type: RoundType struct that indicates what type of move (single, double, etc).
    hand: Hands struct that indicates what type of hand if it is a hand."""

    def __init__(self, cards=torch.zeros(52), type=RoundType.NONE, hand=Hands.NONE):
        self.cards = cards
        self.type = type
        self.hand = hand

    @abstractmethod
    def play(self, player, debug, is_first_move):
        pass

    def clone(self):
        self.cards = self.cards.clone()
        return self

    def to(self, device):
        self.cards = self.cards.clone().to(device)
        return self


class Pass(Action):
    def play(self, player, debug, is_first_move):
        player.passed = True
        if debug:
            print(f"Player {player.number} has passed")


class PlayCards(Action):
    def play(self, player, debug, is_first_move):
        game = player.game
        if debug:
            print("Played the following cards:")
            print_cards(self.cards)
            self.isValid(self.cards, player, game, is_first_move)
        if game.round_type == RoundType.NONE:
            game.round_type = RoundType(torch.sum(self.cards).item())

        player.cards -= self.cards
        game.played_cards[player.number] += self.cards
        game.prev_play = self.cards
        game.prev_player = player
        if debug:
            print(f"Previous player is now player {player.number}.")

        if not torch.any(player.cards):
            if debug:
                print(f"Congratulations, {player} has won!")
            game.finish(player)

    def isValid(self, cards, player, game, is_first_move):
        if is_first_move and not cards[0]:
            raise ValueError("The first move must contain three of clubs!")
        self.ValidCards(cards, player, game)
        self.ValidType(cards, player, game)
        self.ValidRules(cards, player, game)

    def ValidCards(self, cards, player, game):
        invalid_cards = torch.logical_and((1 - player.cards), self.cards)
        if torch.any(invalid_cards):
            invalid_cards_idxs = torch.nonzero(invalid_cards).flatten()
            raise ValueError(
                f"You don't have {idx_to_card(invalid_cards_idxs[0].item())}!"
            )

    def ValidType(self, cards, player, game):
        type = RoundType(torch.sum(cards).item())
        if game.round_type == RoundType.NONE or type == game.round_type:
            return
        raise ValueError(f"You should play a {game.round_type}, not {type}!")

    def ValidRules(self, cards, player, game):
        round_type = game.round_type
        if round_type == RoundType.PAIRS or round_type == RoundType.TRIPLES:
            self.ValidPairsOrTriples(cards, round_type)
            self.ValidValue(cards, player, game)
        if round_type == RoundType.HANDS:
            hand = self.ValidHands(cards, player, game)
            self.ValidHandValue(cards, player, game, hand)

    def ValidPairsOrTriples(self, cards, round_type):
        card_idxs = torch.nonzero(cards).flatten()
        card_vals = torch.div(card_idxs, 4, rounding_mode="trunc")
        if not torch.all(card_vals == card_vals[0]):
            raise ValueError(f"This is not a valid {round_type}!")

    def ValidValue(self, cards, player, game):
        if game.prev_play is not None:
            curr_max = torch.nonzero(cards).flatten()[-1].item()
            prev_max = torch.nonzero(game.prev_play).flatten()[-1].item()
            if curr_max < prev_max:
                raise ValueError(
                    f"The highest card in the previous move was {idx_to_card(prev_max)}, and you played {idx_to_card(curr_max)}!"
                )

    def ValidHands(self, cards, player, game):
        # Determine what hand it is
        straight = DetermineIfStraight(cards)
        flush = DetermineIfFlush(cards)
        most_frequent_count = DetermineMostFrequentCount(cards)

        if straight and flush:
            hand = Hands.STRAIGHT_FLUSH
        elif most_frequent_count == 4:
            hand = Hands.FOUR_OF_A_KIND
        elif most_frequent_count == 3 and torch.nonzero(
            count_cards_per_value(cards)
        ).flatten().shape == (2,):
            hand = Hands.FULL_HOUSE
        elif flush:
            hand = Hands.FLUSH
        elif straight:
            hand = Hands.STRAIGHT
        else:
            raise ValueError("This is not a valid hand!")

        # Determine if the hand can be played
        if game.hand_type:
            if hand < game.hand_type:
                raise ValueError(
                    f"The last move was a {game.hand_type} which is higher than your {hand}!"
                )

        # If it's a new type of hand, reset the previous play
        if hand != game.hand_type:
            game.prev_play = None

        game.hand_type = hand

        return hand

    def ValidHandValue(self, cards, player, game, hand):
        if (
            hand == Hands.STRAIGHT
            or hand == Hands.FLUSH
            or hand == Hands.STRAIGHT_FLUSH
        ):
            self.ValidValue(cards, player, game)

        elif game.prev_play is not None:
            curr_cards_per_value = count_cards_per_value(cards)
            prev_cards_per_value = count_cards_per_value(game.prev_play)
            value = torch.argmax(curr_cards_per_value)
            prev_value = torch.argmax(prev_cards_per_value)
            if value < prev_value:
                raise ValueError(
                    f"The last move was a {game.round_type} of {prev_value}, which is higher than your {value}!"
                )


def DetermineIfStraight(cards):
    values = torch.div(torch.nonzero(cards).flatten(), 4, rounding_mode="trunc").cpu()
    return torch.all(values == torch.arange(min(values), min(values) + 5))


def DetermineIfFlush(cards):
    suits = torch.nonzero(cards).flatten() % 4
    return torch.all(suits == suits[0])


def DetermineMostFrequentCount(cards):
    counts = count_cards_per_value(cards)
    return torch.max(counts)
