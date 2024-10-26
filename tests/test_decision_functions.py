import torch

from pusoy.utils import Hands, indexes_to_one_hot
from pusoy.decision_module import parsing_functions
from pusoy.decision_module.selection_functions import selection_function_eval


class TestFindStraight:
    def test_find_straight_basic(self):
        card_list = indexes_to_one_hot(52, torch.arange(21, 41, 4))

        action = parsing_functions.find_best_straight(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(20, 40, 4)),
            hand_type=Hands.STRAIGHT,
            is_pending=False,
            is_first_move=False,
            selection_function=selection_function_eval
        )

        assert torch.all(action.cards == card_list)

    def test_find_straight_too_low(self):
        card_list = indexes_to_one_hot(52, torch.arange(21, 41, 4))

        action = parsing_functions.find_best_straight(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.STRAIGHT,
            is_pending=False,
            is_first_move=False,
            selection_function=selection_function_eval
        )

        assert action is None

    def test_find_straight_contested(self):
        card_list = indexes_to_one_hot(52, torch.arange(21, 45, 4))

        action = parsing_functions.find_best_straight(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.STRAIGHT,
            is_pending=False,
            is_first_move=False,
            selection_function=selection_function_eval
        )

        assert torch.all(
            action.cards == indexes_to_one_hot(52, torch.arange(25, 45, 4))
        )


class TestFindFlush:
    def test_find_flush_basic(self):
        card_list = indexes_to_one_hot(52, [1, 9, 21, 25, 41])

        action = parsing_functions.find_best_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(20, 40, 4)),
            hand_type=Hands.FLUSH,
            is_pending=False,
            is_first_move=False,
            selection_function=selection_function_eval
        )

        assert torch.all(action.cards == card_list)

    def test_find_flush_too_low(self):
        card_list = indexes_to_one_hot(52, [1, 9, 21, 25, 37])

        action = parsing_functions.find_best_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.FLUSH,
            is_pending=False,
            is_first_move=False,
            selection_function=selection_function_eval
        )

        assert action is None

    def test_find_flush_contested(self):
        card_list = indexes_to_one_hot(52, [1, 9, 21, 25, 37, 2, 6, 10, 14, 50])

        action = parsing_functions.find_best_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.FLUSH,
            is_pending=False,
            is_first_move=False,
            selection_function=selection_function_eval
        )

        assert torch.all(action.cards == indexes_to_one_hot(52, [2, 6, 10, 14, 50]))
