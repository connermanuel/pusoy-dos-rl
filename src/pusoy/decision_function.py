from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F

from pusoy.action import Pass, PlayCards
from pusoy.constants import DEVICE
from pusoy.models import PusoyModel, create_input_tensor, get_probs_from_logits
from pusoy.utils import (
    Hands,
    RoundType,
    card_exists_per_value,
    count_cards_per_suit,
    count_cards_per_value,
    indexes_to_one_hot,
    print_cards,
    string_to_card,
)


class DecisionFunction(ABC):
    """
    Defines the process for making a decision, given the input state.
    """

    def __init__(self):
        pass

    @abstractmethod
    def play(
        self,
        player_no,
        card_list,
        round_type,
        hand_type,
        prev_play,
        prev_player,
        played_cards,
        is_first_move,
    ):
        pass


class Interactive(DecisionFunction):
    def __init__(self, device=DEVICE):
        self.device = device

    def play(
        self,
        player_no,
        card_list,
        round_type,
        hand_type,
        prev_play,
        prev_player,
        played_cards,
        is_first_move,
    ):
        """Returns a 52-len binary tensor of played cards or a pass action."""
        print(f"\nNow playing: Player {player_no}")
        print("\n")
        while True:
            input_str = input(
                (
                    "What action would you like to take?\n"
                    'For a list of available cards, type "show".\n'
                    'To see the round type, type "round".\n'
                    'To see the previous play, type "prev".'
                )
            )

            if input_str == "pass":
                return Pass()

            elif input_str == "show":
                print_cards(card_list)

            elif input_str == "round":
                print(round_type)
                if round_type.value == 5:
                    print(hand_type)

            elif input_str == "prev":
                print_cards(prev_play)

            else:
                cards = torch.zeros(52)
                cards_strs = input_str.split()
                try:
                    for card_str in cards_strs:
                        card = string_to_card(card_str)
                        cards[card.__hash__()] = 1
                    return PlayCards(cards)
                except ValueError:
                    print("Sorry, I couldn't quite understand that.")


class Neural(DecisionFunction):
    """
    Makes decisions based on a neural network's parameters.
    Input of the NN should have the following parameters:
      5 - round type
      5 - hand type
      4 - curr. player
      4 - previous player
      52 - card list
      52 - previously played round
      204 - cards previously played by all players in order from 0 to 3.
    for a total of 330 input parameters.
    Output should have the following parameters, all passed through a sigmoid function:
      52 - (one for each card)
      5 - one for each round type (pass, singles, doubles, triples, hands)
      5 - one for each hand type (straight, flush, full house, four, straight flush)
      for a total of 62 output parameters.
    Stores a list of instances, which are tuples of (input, action) pairs for training.
    """

    model: PusoyModel
    action_funcs = [
        return_pass,
        find_best_single,
        find_best_pair,
        find_best_triple
    ]
    hand_funcs = [
        find_best_straight,
        find_best_flush,
        find_best_full_house,
        find_best_four_hand,
        find_best_straight_flush,
    ]

    def __init__(
        self,
        model: PusoyModel,
        device: torch.device = DEVICE,
        eps: float = 0,
        debug: bool = False,
    ):
        self.model = model.to(device)
        self.instances = {"inputs": [], "actions": []}
        self.device = device
        self.eps = eps
        self.debug = debug

    def selection_function(self, probs, num_samples) -> torch.Tensor:
        """
        Given a tensor of probabilities, return an ordered tensor of indices.
        During training, the selection function is a multinomial distribution.
        During evaluation, the selection function is a maximum function."""

        return torch.topk(input=probs, k=num_samples)[1]

    def play(
        self: DecisionFunction,
        player_no: int,
        card_list: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
        prev_play: torch.Tensor,
        prev_player: int,
        played_cards: list[torch.Tensor],
        is_first_move: bool,
    ):
        # Process player's cardlist, previous round, and previously played cards.
        input = create_input_tensor(
            card_list,
            prev_play,
            played_cards,
            round_type,
            hand_type,
            player_no,
            prev_player,
        )

        # Feed input through NN, and filter output by available cards
        logits = self.model.act(input.to(self.device))
        card_probs, action_probs, hand_probs = get_probs_from_logits(
            logits, card_list, round_type, hand_type
        )

        # Filter the possible round types based on the current round
        is_pending = round_type is RoundType.NONE
        action_funcs = self.get_action_funcs(
            action_probs,
            hand_probs,
            round_type,
            hand_type,
        )

        for action_func in action_funcs:
            action = action_func(
                card_probs.clone(),
                card_list.clone(),
                prev_play,
                hand_type,
                is_pending,
                is_first_move,
            )
            if action:
                self.instances["inputs"].append(input)
                self.instances["actions"].append(action)
                return action

        raise ValueError("No possible actions found!")

    def get_action_funcs(
        self,
        action_probs: torch.Tensor,
        hand_probs: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
    ) -> list[Callable]:
        """
        Returns a list of action-generating functions in order based on the
        predicted logits for different actions and hand types.

        Args:
            action_probs: Probabilities for taking each action
            hand_probs: Probabilities for playing each hand
            round_type: Round being played
            hand_type: Previous hand played

        Returns:
            An ordered list of action generating functions.
        """
        action_mask = round_type.to_tensor(dtype=torch.bool)
        action_mask[1:] ^= action_mask[0]
        action_mask[0] ^= True

        action_funcs = []
        base_action_order = self.selection_function(action_probs, 5)
        action_order = [i for i in base_action_order if action_mask[i]]

        for i in action_order:
            if i == 4:
                action_funcs.extend(self.get_hand_funcs(hand_probs, hand_type))
            else:
                action_funcs.append(self.action_funcs[i])

        return action_funcs

    def get_hand_funcs(
        self, hand_probs: torch.Tensor, hand_type: Hands
    ) -> list[Callable]:
        """
        Returns a list of action-generating functions that play different hands
        in order based on the predicted logits.

        Args:
            hand_probs: Probabilities for playing each hand
            hand_type: Previous hand played

        Returns:
            An ordered list of action generating functions for playing hands.
        """
        min_hand = max(hand_type.value - 1, 0)
        masked_funcs = self.hand_funcs[min_hand:]
        masked_vals = hand_probs[min_hand:]

        action_order = self.selection_function(masked_vals, masked_vals.shape[0])
        return [masked_funcs[i] for i in action_order]

    def clear_instances(self):
        self.instances = []


class TrainingDecisionFunction(Neural):
    def selection_function(self: Neural, probs: torch.Tensor, num_samples: int):
        """Use multinomial selection for training."""
        if self.debug:
            print(probs)
        return torch.multinomial(probs, num_samples)

import torch

from pusoy.action import Pass, PlayCards
from pusoy.utils import Hands, RoundType, card_exists_per_value, count_cards_per_suit, count_cards_per_value, indexes_to_one_hot


def find_best_single(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:
    """
    Selects a single card to play based on probabilities.
    The first move must always be the three of clubs.
    """

    if not is_pending:
        mask = torch.arange(52) > torch.nonzero(prev_play).flatten()[0].item()
        card_list = card_list * mask

    if not torch.any(card_list):
        return None

    best_idx = 0 if is_first_move else selection_function(card_probs, 1)
    cards = indexes_to_one_hot(52, [best_idx])

    return PlayCards(cards, RoundType.SINGLES, Hands.NONE)

def find_best_pair(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:
    return find_best_k_of_a_number(
        2, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    )

def find_best_triple(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:
    return find_best_k_of_a_number(
        3, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    )

def find_best_four(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:
    return find_best_k_of_a_number(
        4, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    )

def find_best_k_of_a_number(
    k: int,
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:
    """
    Finds the best move that plays k of a number. Used for finding pairs,
    triples, and four of a number. The first move must always be of threes, and
    must always contain the three of clubs.

    This does not complete a five card hand - it will only ever select exactly k of
    a number.
    """
    # If k=1, redirect to find_best_single:
    if k == 1:
        return find_best_single(
            card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
        )

    # If first move, force selection of 3C, then select the rest of the cards
    if is_first_move:
        card_list[0] = 0
        card_list[4:] = 0
        card_probs = card_probs * card_list
        other_cards = find_best_k_of_a_number(
            k - 1, card_probs, card_list, prev_play, hand_type, is_pending, False
        )
        if not other_cards:
            return None
        cards = other_cards.cards
        cards[0] = 1
        return PlayCards(cards, RoundType(k), Hands.NONE)

    # Finds all k-counts that you have
    card_counts = count_cards_per_value(card_list)
    valid = card_counts >= k

    # Finds the valid list of highest cards you can play
    if not is_pending:
        prev_highest = get_prev_highest(prev_play)
        exceeds_previous = card_list * (torch.arange(52) > prev_highest)
        valid = valid * card_exists_per_value(exceeds_previous)

    valid_values = torch.nonzero(valid).flatten()
    if len(valid_values) == 0:
        return None

    # Generate a candidate list of k-counts and find the best option
    moves = []
    scores = []
    for value in valid_values:
        card_probs_with_value = card_probs[value * 4 : (value + 1) * 4]
        top_idxs = selection_function(card_probs_with_value, k)
        top_vals = card_probs_with_value[top_idxs]
        scores.append(torch.sum(top_vals))
        moves.append(top_idxs + (value * 4))

    best_move_idx = selection_function(torch.Tensor(scores), 1)[0]
    best_move = moves[best_move_idx]

    cards = indexes_to_one_hot(52, best_move)
    return PlayCards(cards, RoundType(k), Hands.NONE)

def find_best_straight(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:

    if is_first_move:
        # If first move, you must play 3C-4-5-6-7
        card_list[20:] = 0
        card_list[1:4] = 0

    numbers_with_cards = card_exists_per_value(card_list)
    valid = torch.stack(
        [
            numbers_with_cards[0:9],
            numbers_with_cards[1:10],
            numbers_with_cards[2:11],
            numbers_with_cards[3:12],
            numbers_with_cards[4:13],
        ]
    ).all(dim=0)

    valid_last_card_list = card_list
    if not is_pending:
        prev_highest = get_prev_highest(prev_play)
        valid_last_card_list = card_list * (torch.arange(52) > prev_highest)
        valid = valid * card_exists_per_value(valid_last_card_list)[4:13]

    if not valid.any():
        return None

    max_values, max_idxs = (card_probs * card_list).reshape(13, 4).max(dim=1)
    last_max_values, last_max_idxs = (
        (card_probs * valid_last_card_list).reshape(13, 4).max(dim=1)
    )

    valid_idxs = torch.nonzero(valid)

    # Generate a score for each possible straight.
    # The score is defined as the max value for the bottom 4 cards, and the max
    # value for the 5th card from the options that trump the highest card available.
    scores = (
        torch.sum(max_values[valid_idxs + torch.arange(4)], axis=1)
        + last_max_values[valid_idxs.flatten() + 4]
    )
    best_option_value = valid_idxs[selection_function(scores, 1)].item()

    best_idxs = torch.zeros(5).to(torch.long)
    best_idxs[:4] = max_idxs[best_option_value : best_option_value + 4] + (
        torch.arange(best_option_value, best_option_value + 4) * 4
    )
    best_idxs[4] = last_max_idxs[best_option_value + 4] + (
        (best_option_value + 4) * 4
    )

    cards = indexes_to_one_hot(52, best_idxs)
    return PlayCards(cards, RoundType.HANDS, Hands.STRAIGHT)

def find_best_flush(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:

    if hand_type.value < 2:
        is_pending = True

    card_probs = card_probs * card_list
    cards_per_suit = count_cards_per_suit(card_list)
    contains_flush = cards_per_suit >= 5
    if is_first_move:
        # If first move, you can only play clubs!
        contains_flush[1:] = 0

    valid_last_cards = torch.ones(52).to(torch.bool)
    if not is_pending:
        prev_highest = get_prev_highest(prev_play)
        valid_last_cards[torch.arange(52) < prev_highest] = False

    scores = [1e-9]
    options = [None]

    for suit, suit_bool in enumerate(contains_flush):
        if suit_bool:
            output_in_suit = card_probs[suit::4].clone()
            if is_first_move:
                # If first move, you have to play 3C!
                first_val = torch.tensor([output_in_suit[0]])
                output_in_suit[0] = 0
                top_idxs = selection_function(output_in_suit, 4)
                top_vals = output_in_suit[top_idxs]
                top_vals = torch.cat([top_vals, first_val])
                top_idxs = torch.cat([top_idxs, torch.tensor([0])])
            elif not is_pending:
                valid_last_cards_in_suit = (
                    output_in_suit * valid_last_cards[suit::4]
                )
                if not torch.any(valid_last_cards_in_suit):
                    continue
                last_card_idx = selection_function(valid_last_cards_in_suit, 1)
                last_card_val = valid_last_cards_in_suit[last_card_idx]
                output_in_suit[last_card_idx] = 0
                top_idxs = selection_function(output_in_suit, 4)
                top_vals = output_in_suit[top_idxs]
                top_vals = torch.cat([top_vals, last_card_val])
                top_idxs = torch.cat([top_idxs, last_card_idx])
            else:
                # Otherwise, just choose 5 of whatever you have
                top_idxs = selection_function(output_in_suit, 5)
                top_vals = output_in_suit[top_idxs]
            score = torch.sum(top_vals)
            scores.append(score)
            options.append(top_idxs * 4 + suit)

    best_option = options[selection_function(torch.Tensor(scores), 1)]

    if best_option is not None:
        tensor = torch.zeros(52)
        tensor[best_option] = 1
        best_option = PlayCards(tensor, RoundType.HANDS, Hands.FLUSH)

    return best_option

def find_best_full_house(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:

    if hand_type.value < 3:
        is_pending = True

    if not is_pending:
        cards_per_value = count_cards_per_value(prev_play)
        _, value_of_triple = torch.max(cards_per_value, dim=0)
        mask = torch.zeros(52)
        mask[value_of_triple * 4 : (value_of_triple + 1) * 4] = 1
        prev_play = prev_play * mask

    best_triple = find_best_triple(
        card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    )

    if best_triple is None:
        return None
    card_list = card_list * (1 - best_triple.cards)
    card_probs[:52] = card_probs[:52] * card_list

    # Edge case: the first triple is 3S 3H 3D, and it is the first move
    if is_first_move and torch.all(best_triple.cards[1:4]):
        best_triple.cards[3] = 0
        best_triple.cards[0] = 1
    # If it is the first move and you already included it in the triple,
    # no need to include it in the pair!
    if best_triple.cards[0]:
        is_first_move = 0
    # Pair has is_pending enabled, because you can do any pair
    best_pair = find_best_pair(
        card_probs, card_list, prev_play, hand_type, 1, is_first_move
    )
    if best_pair is None:
        return None

    return PlayCards(
        best_triple.cards + best_pair.cards, RoundType.HANDS, Hands.FULL_HOUSE
    )

def find_best_four_hand(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:

    if hand_type.value < 4:
        is_pending = True

    if not is_pending:
        cards_per_value = count_cards_per_value(prev_play)
        _, value_of_triple = torch.max(cards_per_value, dim=0)
        mask = torch.zeros(52)
        mask[value_of_triple * 4 : (value_of_triple + 1) * 4] = 1
        prev_play = prev_play * mask

    best_four = find_best_four(
        card_probs, card_list, prev_play, hand_type, is_pending, 0
    )

    if best_four is None:
        return None

    # If it is the first move and you already included it in the four,
    # no need to include it in the single!
    if best_four.cards[0]:
        is_first_move = 0
    card_list = card_list * (1 - best_four.cards)
    card_probs = card_probs * card_list
    best_single = find_best_single(
        card_probs, card_list, prev_play, hand_type, 1, is_first_move
    )
    if best_single is None:
        return None

    return PlayCards(
        best_four.cards + best_single.cards, RoundType.HANDS, Hands.FOUR_OF_A_KIND
    )

def find_best_straight_flush(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: callable
) -> PlayCards | None:
    # If it is the first move, only possible straight flush is 3C 4C 5C 6C 7C
    if is_first_move:
        if torch.all(card_list[0:20:4]):
            cards = torch.zeros(52)
            cards[0:20:4] = 1
            return PlayCards(cards, RoundType.HANDS, Hands.STRAIGHT_FLUSH)
        return None

    if hand_type.value < 5:
        is_pending = True

    contains_flush = count_cards_per_suit(card_list) >= 5

    scores = [1e-9]
    actions = [None]

    for suit, suit_bool in enumerate(contains_flush):
        if suit_bool:
            mask = torch.zeros(52)
            mask[suit::4] = 1
            masked_card_list = card_list * mask
            masked_output = card_probs * masked_card_list
            best_straight = find_best_straight(
                masked_output,
                masked_card_list,
                prev_play,
                hand_type,
                is_pending,
                is_first_move,
            )

            if best_straight is not None:
                score = torch.sum(card_probs[best_straight.cards.to(torch.bool)])
                scores.append(score)
                actions.append(best_straight)

    best_action = actions[selection_function(torch.Tensor(scores), 1)]
    if best_action is not None:
        best_action.hands = Hands.STRAIGHT_FLUSH
    return best_action

def return_pass(
    output, card_list, prev_play, hand_type, is_pending, is_first_move
):
    return Pass(cards=torch.zeros(52))

def get_prev_highest(prev_play: torch.Tensor) -> int:
    """Gets the highest index of the previous play"""
    return torch.nonzero(prev_play).flatten()[-1].item()