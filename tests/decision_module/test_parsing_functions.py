import torch

from pusoy.action import Pass
from pusoy.utils import Hands, indexes_to_one_hot, card_names_to_card_list
from pusoy.decision_module import parsing_functions
from pusoy.decision_module.selection_functions import selection_function_eval


class TestFindBestSingle:
    def test_find_best_single_basic(self):
        card_list = card_names_to_card_list(["3C"])
        action = parsing_functions.find_best_single(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert torch.all(action.cards == card_list)

    def test_find_best_single_no_valid_move(self):
        card_list = torch.zeros(52)
        action = parsing_functions.find_best_single(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None

    def test_find_best_single_no_valid_first_move(self):
        card_list = card_names_to_card_list(["3S"])
        action = parsing_functions.find_best_single(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None


class TestFindBestPair:
    def test_find_best_pair_basic(self):
        card_list = card_names_to_card_list(["3C", "3S"])
        action = parsing_functions.find_best_pair(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert torch.all(action.cards == card_list)

    def test_find_best_pair_no_valid_move(self):
        card_list = torch.zeros(52)
        action = parsing_functions.find_best_pair(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None


class TestFindBestTriple:
    def test_find_best_triple_basic(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H"])
        action = parsing_functions.find_best_triple(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert torch.all(action.cards == card_list)

    def test_find_best_triple_no_valid_move(self):
        card_list = card_names_to_card_list(["3C", "3S", "4H"])
        action = parsing_functions.find_best_triple(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None


class TestFindBestFour:
    def test_find_best_four_basic(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H", "3D"])
        action = parsing_functions.find_best_four(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert torch.all(action.cards == card_list)

    def test_find_best_four_no_valid_move(self):
        card_list = torch.zeros(52)
        action = parsing_functions.find_best_four(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None


class TestFindStraight:
    def test_find_straight_basic(self):
        card_list = indexes_to_one_hot(52, torch.arange(21, 41, 4))

        action = parsing_functions.find_best_straight(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(20, 40, 4)),
            hand_type=Hands.STRAIGHT,
            is_first_move=False,
            selection_function=selection_function_eval,
        )

        assert torch.all(action.cards == card_list)

    def test_find_straight_too_low(self):
        card_list = indexes_to_one_hot(52, torch.arange(21, 41, 4))

        action = parsing_functions.find_best_straight(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.STRAIGHT,
            is_first_move=False,
            selection_function=selection_function_eval,
        )

        assert action is None

    def test_find_straight_contested(self):
        card_list = indexes_to_one_hot(52, torch.arange(21, 45, 4))

        action = parsing_functions.find_best_straight(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.STRAIGHT,
            is_first_move=False,
            selection_function=selection_function_eval,
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
            is_first_move=False,
            selection_function=selection_function_eval,
        )

        assert torch.all(action.cards == card_list)

    def test_find_flush_too_low(self):
        card_list = indexes_to_one_hot(52, [1, 9, 21, 25, 37])

        action = parsing_functions.find_best_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.FLUSH,
            is_first_move=False,
            selection_function=selection_function_eval,
        )

        assert action is None

    def test_find_flush_contested(self):
        card_list = indexes_to_one_hot(52, [1, 9, 21, 25, 37, 2, 6, 10, 14, 50])

        action = parsing_functions.find_best_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=indexes_to_one_hot(52, torch.arange(24, 44, 4)),
            hand_type=Hands.FLUSH,
            is_first_move=False,
            selection_function=selection_function_eval,
        )

        assert torch.all(action.cards == indexes_to_one_hot(52, [2, 6, 10, 14, 50]))


class TestFindBestFullHouse:
    def test_find_best_full_house_basic(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H", "4C", "4D"])
        action = parsing_functions.find_best_full_house(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is not None

    def test_find_best_full_house_no_valid_move(self):
        card_list = torch.zeros(52)
        action = parsing_functions.find_best_full_house(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None

    def test_find_best_full_house_against_straight(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H", "4C", "4D"])
        prev_play = card_names_to_card_list(["5C", "6S", "7H", "8C", "9D"])
        action = parsing_functions.find_best_full_house(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=prev_play,
            hand_type=Hands.STRAIGHT,
            is_first_move=False,
            selection_function=selection_function_eval,
        )
        assert torch.all(action.cards == card_list)

    def test_find_best_full_house_against_flush(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H", "4C", "4D"])
        prev_play = card_names_to_card_list(["5C", "6C", "7C", "8C", "10C"])
        action = parsing_functions.find_best_full_house(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=prev_play,
            hand_type=Hands.FLUSH,
            is_first_move=False,
            selection_function=selection_function_eval,
        )
        assert torch.all(action.cards == card_list)


class TestFindBestFourHand:
    def test_find_best_four_hand_basic(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H", "3D", "4C"])
        action = parsing_functions.find_best_four_hand(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is not None

    def test_find_best_four_hand_no_valid_move(self):
        card_list = card_names_to_card_list(["3C", "3S", "3H", "4C", "4D"])
        action = parsing_functions.find_best_four_hand(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None


class TestFindBestStraightFlush:
    def test_find_best_straight_flush_basic(self):
        card_list = indexes_to_one_hot(52, [0, 4, 8, 12, 16])
        action = parsing_functions.find_best_straight_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is not None

    def test_find_best_straight_flush_no_valid_move(self):
        card_list = torch.zeros(52)
        action = parsing_functions.find_best_straight_flush(
            card_probs=torch.rand(52),
            card_list=card_list,
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert action is None


class TestReturnPass:
    def test_return_pass(self):
        action = parsing_functions.return_pass(
            card_probs=torch.rand(52),
            card_list=torch.zeros(52),
            prev_play=torch.zeros(52),
            hand_type=Hands.NONE,
            is_first_move=True,
            selection_function=selection_function_eval,
        )
        assert isinstance(action, Pass)
