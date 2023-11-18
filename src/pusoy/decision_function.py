from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import torch

from pusoy.action import Pass, PlayCards
from pusoy.models import PusoyModel
from pusoy.utils import (
    Hands,
    RoundType,
    card_exists_per_value,
    count_cards_per_suit,
    count_cards_per_value,
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
    def __init__(self, device="cuda"):
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
                """
What action would you like to take?\n
For a list of available cards, type \"show\".\n
To see the round type, type \"round\".\n
To see the previous play, type \"prev\".
"""
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
                except ValueError as ve:
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
      5 - one for each hand type (straight, flush, full house, four of a kind, straight flush)
      for a total of 62 output parameters.
    Stores a list of instances, which are tuples of (input, action) pairs used for training.
    """

    def __init__(self, model: PusoyModel, device="cuda", eps=0, debug=False):
        self.device = device
        self.model = model.to(device)
        self.instances = {"inputs": [], "actions": []}
        self.funcs = [
            self.return_pass,
            self.find_best_single,
            self.find_best_pair,
            self.find_best_triple,
            self.find_best_straight,
            self.find_best_flush,
            self.find_best_full_house,
            self.find_best_four_hand,
            self.find_best_straight_flush,
        ]
        self.device = device
        self.eps = eps
        self.debug = debug
        self.hx = None

    def selection_function(self, probs, num_samples):
        """
        Given a tensor of probabilities, return an ordered tensor of indices.
        During training, the selection function is a multinomial distribution.
        During evaluation, the selection function is a maximum function.
        """
        return torch.topk(input=probs, k=num_samples)[1]

    def play(
        self: DecisionFunction,
        player_no: int,
        card_list: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
        prev_play: torch.Tensor,
        prev_player: int,
        played_cards: List[torch.Tensor],
        is_first_move: bool,
    ):
        """
        Note: all of the selection functions expect card_probs to be on self.device,
        and for card_list to be on CPU.
        """
        # Get the LSTM output and hidden states
        out, hx = self.model.get_hidden_state_from_input(
            player_no,
            card_list,
            round_type,
            hand_type,
            prev_play,
            prev_player,
            played_cards,
            self.hx,
        )
        self.hx = hx
        card_list_embeddings = self.model.get_card_list_embeddings(card_list)
        action_probabilties = self.model.get_action_probabilities(out)

        round_mask = round_type.to_tensor(dtype=torch.bool)
        round_mask[1:] ^= round_mask[0]
        round_mask[0] ^= True

        hand_mask = hand_type.to_tensor().cumsum(0).type(torch.bool)
        if round_type == RoundType.NONE:
            hand_mask = hand_mask + True

        mask = torch.cat([round_mask[:-1], hand_mask]).to(self.device)

        action_probabilties = (action_probabilties + self.eps) * mask
        action_order = self.selection_function(action_probabilties, 9)
        card_idxs = torch.nonzero(card_list).flatten()

        for idx in action_order:
            creation_func = self.funcs[idx]
            if idx == 0:
                action = Pass()
            else:
                card_probabiltiies_flat = self.model.get_card_probabilities(
                    out, card_list_embeddings, idx
                )
                card_probabilities = torch.zeros(52).to(self.device)
                card_probabilities[card_idxs] = card_probabiltiies_flat + 1e-16

                assert torch.eq(
                    card_probabilities, card_probabilities * card_list.to(self.device)
                ).all()

                action = creation_func(
                    card_probabilities,
                    card_list.clone(),
                    prev_play,
                    hand_type.to_tensor(),
                    ~round_mask[0],
                    is_first_move,
                )
            if action:
                self.instances["inputs"].append(
                    deepcopy({
                        "player_no": player_no,
                        "card_list": card_list,
                        "round_type": round_type,
                        "hand_type": hand_type,
                        "prev_play": prev_play,
                        "prev_player": prev_player,
                        "played_cards": played_cards,
                    })
                )
                self.instances["actions"].append(action)
                return action

        raise ValueError("No possible actions found!")

    def find_best_single(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        if not torch.any(card_list):
            return None
        cards = torch.zeros(52)
        if is_first_move:
            if not card_list[0]:
                return None
            best_idx = 0
        else:
            if not is_pending:
                mask = torch.arange(52) > torch.nonzero(prev_play).flatten()[0].item()
                card_list = card_list * mask
                if not torch.any(card_list):
                    return None
            card_probs = card_probs
            best_idx = self.selection_function(card_probs, 1)
        cards[best_idx] = 1
        return PlayCards(cards, RoundType.SINGLES, Hands.NONE)

    def find_best_pair(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        return self.find_best_k_of_a_number(
            2, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
        )

    def find_best_triple(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        return self.find_best_k_of_a_number(
            3, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
        )

    def find_best_four(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        return self.find_best_k_of_a_number(
            4, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
        )

    def find_best_k_of_a_number(
        self, k, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        if is_first_move:
            card_list[4:] = 0
            card_probs = card_probs * card_list.to(self.device)
            other_cards = self.find_best_k_of_a_number(
                k - 1, card_probs, card_list, prev_play, hand_type, is_pending, False
            )
            if other_cards is not None:
                cards = other_cards.cards
                cards[0] = 1
                return PlayCards(cards, RoundType(k), Hands.NONE)
            return None

        # If k=1, redirect to find_best_single:
        if k == 1:
            return self.find_best_single(
                card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
            )

        # Finds all k-counts that you have
        card_counts = count_cards_per_value(card_list)
        has_a_pair_per_number = card_counts >= k

        # Finds the valid list of highest cards you can play
        valid = count_cards_per_value(card_list).to(torch.bool)
        if not is_pending:
            idx_highest = torch.div(
                torch.nonzero(prev_play).flatten()[-1].item(), 4, rounding_mode="trunc"
            )
            valid = valid * (torch.arange(13) > idx_highest)

        valid = valid * has_a_pair_per_number
        idx_options = torch.nonzero(valid).flatten()

        # Out of all valid k-counts, finds the best option
        moves = [None]
        scores = [1e-9]
        for option in idx_options:
            range = card_probs[option * 4 : (option + 1) * 4]
            try:
                top_idxs = self.selection_function(range, k)
            except RuntimeError as e:
                raise e
            top_vals = range[top_idxs]
            score = torch.sum(top_vals)
            scores.append(score)
            moves.append(top_idxs + option * 4)

        best_move_idx = self.selection_function(torch.Tensor(scores), 1)[0]
        best_move = moves[best_move_idx]

        if best_move is not None:
            tensor = torch.zeros(52)
            tensor[best_move] = 1
            best_move = PlayCards(tensor, RoundType(k), Hands.NONE)

        return best_move

    def find_best_straight(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        if is_first_move:
            # If first move, the only straights you can play are 3-4-5-6-7, and you must use 3C!
            card_list[20:] = 0
            card_list[1:4] = 0
            card_probs = card_probs * card_list.to(self.device)
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
            prev_highest = torch.nonzero(prev_play).flatten()[-1].item()
            valid_last_card_list = valid_last_card_list * (
                torch.arange(52) > prev_highest
            )
            valid = valid * card_exists_per_value(valid_last_card_list)[4:13]

        if not valid.any():
            return None

        max_values, max_idxs = card_probs.reshape(13, 4).max(dim=1)
        max_idxs = max_idxs.cpu()
        last_max_values, last_max_idxs = (
            (card_probs * valid_last_card_list.to(self.device))
            .reshape(13, 4)
            .max(dim=1)
        )

        valid_idxs = torch.nonzero(valid).flatten()
        if valid_idxs.nelement() == 0:
            return None

        scores = (
            torch.sum(max_values[valid_idxs.reshape(-1, 1) + torch.arange(4)], axis=1)
            + last_max_values[valid_idxs]
        )
        best_option = self.selection_function(scores, 1).cpu()
        best_option = valid_idxs[best_option].item()

        best_idxs = torch.zeros(5).to(torch.long)
        best_idxs[:4] = max_idxs[best_option : best_option + 4] + (
            torch.arange(best_option, best_option + 4) * 4
        )
        best_idxs[4] = last_max_idxs[best_option + 4] + ((best_option + 4) * 4)

        tensor = torch.zeros(52)
        tensor[best_idxs] = 1
        return PlayCards(tensor, RoundType.HANDS, Hands.STRAIGHT)

    def find_best_flush(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        if hand_type[0].item():
            is_pending = 1
        cards_per_suit = count_cards_per_suit(card_list)
        contains_flush = cards_per_suit >= 5
        if is_first_move:
            # If first move, you can only play clubs!
            contains_flush[1:] = 0

        valid_last_cards = torch.ones(52).to(torch.bool)
        if not is_pending:
            max_value_from_prev_play = torch.nonzero(prev_play).flatten()[-1]
            valid_last_cards[torch.arange(52) < max_value_from_prev_play] = False

        scores = [1e-9]
        options = [None]

        for suit, suit_bool in enumerate(contains_flush):
            if suit_bool:
                output_in_suit = card_probs[suit:52:4].clone().cpu()
                if is_first_move:
                    # If first move, you have to play 3C!
                    first_val = torch.tensor([output_in_suit[0]])
                    output_in_suit[0] = 0
                    top_idxs = self.selection_function(output_in_suit, 4)
                    top_vals = output_in_suit[top_idxs]
                    top_vals = torch.cat([top_vals, first_val])
                    top_idxs = torch.cat([top_idxs, torch.tensor([0])])
                elif not is_pending:
                    # If you have to beat another flush, your best card needs to trump theirs!
                    valid_last_cards_in_suit = (
                        output_in_suit * valid_last_cards[suit::4]
                    )
                    if not torch.any(valid_last_cards_in_suit):
                        return None
                    last_card_idx = self.selection_function(valid_last_cards_in_suit, 1)
                    last_card_val = valid_last_cards_in_suit[last_card_idx]
                    output_in_suit[last_card_idx] = 0
                    top_idxs = self.selection_function(output_in_suit, 4)
                    top_vals = output_in_suit[top_idxs]
                    top_vals = torch.cat([top_vals, last_card_val])
                    top_idxs = torch.cat([top_idxs, last_card_idx])
                else:
                    # Otherwise, just choose 5 of whatever you have
                    top_idxs = self.selection_function(output_in_suit, 5)
                    top_vals = output_in_suit[top_idxs]
                score = torch.sum(top_vals)
                scores.append(score)
                options.append(top_idxs * 4 + suit)

        best_option = options[self.selection_function(torch.Tensor(scores), 1)]

        if best_option is not None:
            tensor = torch.zeros(52)
            tensor[best_option] = 1
            best_option = PlayCards(tensor, RoundType.HANDS, Hands.FLUSH)

        return best_option

    def find_best_full_house(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        if torch.sum(hand_type[:2]).item():
            is_pending = 1
        if not is_pending:
            cards_per_value = count_cards_per_value(prev_play)
            _, value_of_triple = torch.max(cards_per_value, dim=0)
            mask = torch.zeros(52)
            mask[value_of_triple * 4 : (value_of_triple + 1) * 4] = 1
            prev_play = prev_play * mask

        best_triple = self.find_best_triple(
            card_probs, card_list, prev_play, hand_type, is_pending, is_first_move=0
        )  # Even if it is the first move, we should allow the triple to contain other numbers

        if best_triple is None:
            return None
        card_probs = card_probs * (1 - best_triple.cards).to(self.device)
        card_list = card_list * (1 - best_triple.cards)

        # Edge case: the first triple is 3S 3H 3D, and it is the first move (which means we should've included 3C)
        if is_first_move and torch.all(best_triple.cards[1:4]):
            best_triple = self.find_best_triple(
                card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
            )  # If we were going to get 3s anyway, just enforce the first move rule
        # If it is the first move and you already included it in the triple, no need to include it in the pair!
        if best_triple.cards[0]:
            is_first_move = 0
        # Pair has is_pending enabled, because you can do any pair
        best_pair = self.find_best_pair(
            card_probs, card_list, prev_play, hand_type, 1, is_first_move
        )
        if best_pair is None:
            return None

        return PlayCards(
            best_triple.cards + best_pair.cards, RoundType.HANDS, Hands.FULL_HOUSE
        )

    def find_best_four_hand(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        if torch.sum(hand_type[:3]).item():
            is_pending = 1

        if not is_pending:
            cards_per_value = count_cards_per_value(prev_play)
            _, value_of_triple = torch.max(cards_per_value, dim=0)
            mask = torch.zeros(52)
            mask[value_of_triple * 4 : (value_of_triple + 1) * 4] = 1
            prev_play = prev_play * mask

        best_four = self.find_best_four(
            card_probs, card_list, prev_play, hand_type, is_pending, 0
        )

        if best_four is None:
            return None

        # If it is the first move and you already included it in the four, no need to include it in the single!
        if best_four.cards[0]:
            is_first_move = 0
        card_probs = card_probs * (1 - best_four.cards).to(self.device)
        card_list = card_list * (1 - best_four.cards)
        best_single = self.find_best_single(
            card_probs, card_list, prev_play, hand_type, 1, is_first_move
        )
        if best_single is None:
            return None

        return PlayCards(
            best_four.cards + best_single.cards, RoundType.HANDS, Hands.FOUR_OF_A_KIND
        )

    def find_best_straight_flush(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        # If it is the first move, only possible straight flush is 3C 4C 5C 6C 7C
        if is_first_move:
            if torch.all(card_list[0:20:4]):
                cards = torch.zeros(52)
                cards[0:20:4] = 1
                return PlayCards(cards, RoundType.HANDS, Hands.STRAIGHT_FLUSH)
            return None
        if not hand_type[4].item():
            is_pending = 1
        contains_flush = count_cards_per_suit(card_list) >= 5

        scores = [1e-9]
        actions = [None]

        for suit, suit_bool in enumerate(contains_flush):
            if suit_bool:
                mask = torch.zeros(52)
                mask[suit::4] = 1
                masked_card_list = card_list * mask
                masked_output = card_probs * masked_card_list.to(self.device)
                best_straight = self.find_best_straight(
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

        best_action = actions[self.selection_function(torch.Tensor(scores), 1)]
        if best_action is not None:
            best_action.hands = Hands.STRAIGHT_FLUSH
        return best_action

    def return_pass(
        self, card_probs, card_list, prev_play, hand_type, is_pending, is_first_move
    ):
        return Pass(cards=torch.zeros(52))

    def clear_instances(self):
        self.instances = []


class TrainingDecisionFunction(Neural):
    def selection_function(self: Neural, probs: torch.Tensor, num_samples: int):
        if self.debug:
            print(probs)
        return torch.multinomial(probs, num_samples)
