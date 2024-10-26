import torch
from typing import Callable

from pusoy.utils import (
    Hands,
    RoundType,
    card_exists_per_value,
    count_cards_per_suit,
    count_cards_per_value,
    indexes_to_one_hot,
)

# Define the SelectionFunction type
SelectionFunction = Callable[[torch.Tensor, int], int]

def find_best_single(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: SelectionFunction
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
    selection_function: SelectionFunction
) -> PlayCards | None:
    return find_best_k_of_a_number(
        2, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move, selection_function
    )

def find_best_triple(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: SelectionFunction
) -> PlayCards | None:
    return find_best_k_of_a_number(
        3, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move, selection_function
    )

def find_best_four(
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: SelectionFunction
) -> PlayCards | None:
    return find_best_k_of_a_number(
        4, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move, selection_function
    )

def find_best_k_of_a_number(
    k: int,
    card_probs: torch.Tensor,
    card_list: torch.Tensor,
    prev_play: torch.Tensor,
    hand_type: Hands,
    is_pending: bool,
    is_first_move: bool,
    selection_function: SelectionFunction
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
            card_probs, card_list, prev_play, hand_type, is_pending, is_first_move, selection_function
        )

    # If first move, force selection of 3C, then select the rest of the cards
    if is_first_move:
        card_list[0] = 0
        card_list[4:] = 0
        card_probs = card_probs * card_list
        other_cards = find_best_k_of_a_number(
            k - 1, card_probs, card_list, prev_play, hand_type, is_pending, False, selection_function
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
    selection_function: Callable
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
    selection_function: SelectionFunction
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
            score = torch.sum(top_vals).item()
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
    selection_function: SelectionFunction
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
    selection_function: SelectionFunction
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
    selection_function: SelectionFunction
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

def selection_function_train(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Given a tensor of probabilities, return an ordered tensor of indices.
    During training, the selection function is a multinomial distribution.
    """
    return torch.multinomial(probs, num_samples, replacement=True)

def selection_function_eval(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Given a tensor of probabilities, return an ordered tensor of indices.
    During evaluation, the selection function is a maximum function.
    """
    return torch.topk(input=probs, k=num_samples)[1]