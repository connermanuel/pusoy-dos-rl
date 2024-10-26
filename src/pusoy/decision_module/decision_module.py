from abc import ABC, abstractmethod
from typing import Callable

import torch

from pusoy.action import Pass, PlayCards
from pusoy.constants import DEVICE
from pusoy.models import PusoyModel, create_input_tensor, get_probs_from_logits
from pusoy.utils import (
    Hands,
    RoundType,
    string_to_card,
    print_cards
)
from pusoy.decision_module import parsing_functions, selection_functions


class DecisionModule(ABC):
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


class Interactive(DecisionModule):
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


class Neural(DecisionModule):
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
        parsing_functions.return_pass,
        parsing_functions.find_best_single,
        parsing_functions.find_best_pair,
        parsing_functions.find_best_triple
    ]
    hand_funcs = [
        parsing_functions.find_best_straight,
        parsing_functions.find_best_flush,
        parsing_functions.find_best_full_house,
        parsing_functions.find_best_four_hand,
        parsing_functions.find_best_straight_flush,
    ]
    selection_function = staticmethod(selection_functions.selection_function_eval)

    def __init__(
        self,
        model: PusoyModel,
        device: torch.device = DEVICE,
        eps: float = 0,
        debug: bool = False,
    ):
        self.model = model.to(device)
        self.instances: dict[str, list] = {"inputs": [], "actions": []}
        self.device = device
        self.eps = eps
        self.debug = debug

    def play(
        self,
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


class TrainingNeural(Neural):
    selection_function = staticmethod(selection_functions.selection_function_train)

