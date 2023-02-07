from doctest import debug_script
import torch
import torch.nn.functional as F

from pusoy.action import Pass, PlayCards
from pusoy.utils import string_to_card, print_cards, RoundType, Hands, count_cards_per_value, card_exists_per_value, count_cards_per_suit

from abc import ABC, abstractmethod
from typing import List

class DecisionFunction(ABC):
    # What information does the player have in a game of pusoy dos when making a decision?
    # They have: 
    #   which player they are
    #   their list of cards, 
    #   the round type, 
    #   the previously played round,
    #   the previous player,
    #   and the list of cards all persons have played, from 1 to 4.

    def __init__(self):
        pass

    @abstractmethod
    def play(self, player_no, card_list, round_type, hand_type, prev_play, prev_player, played_cards, is_first_move):
        pass

class Interactive(DecisionFunction):
    def play(self, player_no, card_list, round_type, hand_type, prev_play, prev_player, played_cards, is_first_move):
        """Returns a 52-len binary tensor of played cards or a pass action."""
        print(f'\nNow playing: Player {player_no}')
        print('\n')
        while True:
            input_str = input("""
What action would you like to take?\n
For a list of available cards, type \"show\".\n
To see the round type, type \"round\".\n
To see the previous play, type \"prev\".
""")

            if input_str == 'pass':
                return Pass()
            
            elif input_str == 'show':
                print_cards(card_list)
            
            elif input_str == 'round':
                print(round_type)
                if round_type.value == 5:
                    print(hand_type)
            
            elif input_str == 'prev':
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
    def __init__(self, model, device='cuda', eps=0, debug=False):
        self.model = model.to(device)
        self.instances = []
        self.funcs = [self.return_pass, self.find_best_single, self.find_best_pair, self.find_best_triple, self.find_best_hand]
        self.hand_funcs = [
            self.find_best_straight,
            self.find_best_flush,
            self.find_best_full_house,
            self.find_best_four_hand,
            self.find_best_straight_flush
        ]
        self.device = device
        self.eps=eps
        self.debug=debug
    
    def selection_function(self, probs, num_samples):
        """
        Given a tensor of probabilities, return an ordered tensor of indices.
        During training, the selection function is a multinomial distribution.
        During evaluation, the selection function is a maximum function."""
        return torch.topk(input=probs, k=num_samples)[1]

    def play(self: DecisionFunction, player_no: int, card_list: torch.Tensor, round_type: RoundType, 
             hand_type: Hands, prev_play: torch.Tensor, prev_player: int, played_cards: List, is_first_move: bool):
        """
        Args:
        eps -- exploration parameter
        """
        if torch.any(card_list < 0):
            raise ValueError(f"Negative value in card list, {card_list}")
        # Process round type, current player, and previous player
        round_type = round_type.to_tensor(device=self.device)
        hand_type = hand_type.to_tensor(device=self.device)
        if prev_play is None:
            prev_play = torch.zeros(52, device=self.device)
        player_no_vec, prev_player_vec = torch.zeros(4, device=self.device), torch.zeros(4, device=self.device)
        player_no_vec[player_no] = 1
        if prev_player is not None:
            prev_player_vec[prev_player] = 1
        input = torch.cat([round_type.to(torch.float), hand_type.to(torch.float), player_no_vec, prev_player_vec])

        # Process player's cardlist, previously played round, and all previously played cards.
        card_lists = [card_list] + [prev_play] + played_cards
        card_tensors = torch.cat(card_lists)
        input = torch.cat([input, card_tensors])
        self.input = input

        # Feed input through NN, and filter output by available cards
        output = self.model.actor(input)
        output[:52] = F.softmax(output[:52], dim=0)
        output[52:57] = F.softmax(output[52:57], dim=0)
        output[57:] = F.softmax(output[57:], dim=0)

        output = output + self.eps + 1e-8
        output[:52] = output[:52] * card_list

        # Build a queue of potential actions, based on the return values.
        round_type_vals = output[52:57] # pass, single, double, triple, hand
        round_type[1:] ^= round_type[0]
        round_type[0] ^= True
        round_type_vals = round_type_vals * round_type
        action_order = self.selection_function(round_type_vals, 5)

        for idx in action_order:
            creation_func = self.funcs[idx]
            action = creation_func(output.clone(), card_list.clone(), prev_play, hand_type, ~round_type[0], is_first_move)
            if action:
                self.instances.append((input, action))
                return action
        
        raise ValueError('No possible actions found!')
    
    def find_best_single(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        if not torch.any(card_list):
            return None
        cards = torch.zeros(52, device=self.device)
        if is_first_move:
            if not card_list[0]:
                return None
            best_idx = 0
        else:
            if not is_pending:
                mask = torch.arange(52, device=self.device) > torch.nonzero(prev_play).flatten()[0].item()
                card_list = card_list * mask
                if not torch.any(card_list):
                    return None
            output[:52] = output[:52] * card_list
            best_idx = self.selection_function(output[:52], 1)
        cards[best_idx] = 1
        return PlayCards(cards, RoundType.SINGLES, Hands.NONE)

    def find_best_pair(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        return self.find_best_k_of_a_number(2, output, card_list, prev_play, hand_type, is_pending, is_first_move)

    def find_best_triple(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        return self.find_best_k_of_a_number(3, output, card_list, prev_play, hand_type, is_pending, is_first_move)
    
    def find_best_four(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        return self.find_best_k_of_a_number(4, output, card_list, prev_play, hand_type, is_pending, is_first_move)
    
    def find_best_k_of_a_number(self, k, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        if is_first_move:
            card_list[4:] = 0
            other_cards = self.find_best_k_of_a_number(k-1, output, card_list, prev_play, hand_type, is_pending, False)
            if other_cards is not None:
                cards = other_cards.cards
                cards[0] = 1
                return PlayCards(cards, RoundType(k), Hands.NONE)
            return None
        
        # If k=1, redirect to find_best_single:
        if k==1:
            return self.find_best_single(output, card_list, prev_play, hand_type, is_pending, is_first_move)

        # Finds all k-counts that you have
        card_counts = count_cards_per_value(card_list)
        has_a_pair_per_number = card_counts >= k

        # Finds the valid list of highest cards you can play
        valid = count_cards_per_value(card_list).to(torch.bool)
        if not is_pending:
            idx_highest = torch.div(torch.nonzero(prev_play).flatten()[-1].item(), 4, rounding_mode='trunc')
            valid = valid * (torch.arange(13, device=self.device) > idx_highest)
        
        valid = valid * has_a_pair_per_number
        idx_options = torch.nonzero(valid).flatten()

        # Out of all valid k-counts, finds the best option
        moves = [None]
        scores = [1e-9]
        for option in idx_options:
            range = output[option*4:(option+1)*4]
            top_idxs = self.selection_function(range, k)
            top_vals = range[top_idxs]
            score = torch.sum(top_vals)
            scores.append(score)
            moves.append(top_idxs + option * 4)
        
        best_move_idx = self.selection_function(torch.Tensor(scores), 1)[0]
        best_move = moves[best_move_idx]

        if best_move is not None:
            tensor = torch.zeros(52, device=self.device)
            tensor[best_move] = 1
            best_move = PlayCards(tensor, RoundType(k), Hands.NONE)
        
        return best_move

    def find_best_hand(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        hand_values = output[-5:] # straight, flush, full house, fours, straight flush

        # Action queue can only be composed of valid actions from valid round types
        idx_hand_type = torch.nonzero(hand_type).flatten()[0].item()
        masked_funcs = self.hand_funcs[idx_hand_type:]
        masked_vals = hand_values[idx_hand_type:]
        
        action_order = self.selection_function(masked_vals, masked_vals.shape[0])

        for idx in action_order:
            creation_func = masked_funcs[idx]
            action = creation_func(output.clone(), card_list.clone(), prev_play, hand_type, is_pending, is_first_move)
            if action:
                return action            
            
        return None
    
    def find_best_straight(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        if is_first_move:
            # If first move, the only straights you can play are 3-4-5-6-7, and you must use 3C!
            card_list[20:] = 0
            card_list[1:4] = 0
        numbers_with_cards = card_exists_per_value(card_list)
        valid = torch.stack([
            numbers_with_cards[0:9], numbers_with_cards[1:10], numbers_with_cards[2:11], numbers_with_cards[3:12], numbers_with_cards[4:13]
        ]).all(dim=0)
        valid_last_card_list = card_list
        if not is_pending:
            prev_highest = torch.nonzero(prev_play).flatten()[-1].item()
            valid_last_card_list = valid_last_card_list * (torch.arange(52, device=self.device) > prev_highest)
            valid = valid * card_exists_per_value(valid_last_card_list)[4:13]

        if not valid.any():
            return None
        
        max_values, max_idxs = (output[:52] * card_list).reshape(13, 4).max(dim=1)
        last_max_values, last_max_idxs = (output[:52] * valid_last_card_list).reshape(13, 4).max(dim=1)

        valid_idxs = torch.nonzero(valid).flatten()
        if valid_idxs.nelement() == 0:
            return None

        scores = torch.sum(max_values[valid_idxs.reshape(-1, 1) + torch.arange(4, device=self.device)], axis = 1) + last_max_values[valid_idxs]
        best_option = self.selection_function(scores, 1)
        best_option = valid_idxs[best_option].item()

        best_idxs = torch.zeros(5).to(torch.long)
        best_idxs[:4] = max_idxs[best_option:best_option+4] + (torch.arange(best_option, best_option+4, device=self.device) * 4)
        best_idxs[4] = last_max_idxs[best_option+4] + ((best_option+4) * 4)

        tensor = torch.zeros(52, device=self.device)
        tensor[best_idxs] = 1
        return PlayCards(tensor, RoundType.HANDS, Hands.STRAIGHT)

    def find_best_flush(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
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
                output_in_suit = output[suit:52:4].clone()
                if is_first_move:
                    # If first move, you have to play 3C!
                    first_val = torch.tensor([output_in_suit[0]], device=self.device)
                    output_in_suit[0] = 0
                    top_idxs = self.selection_function(output_in_suit, 4)
                    top_vals = output_in_suit[top_idxs]
                    top_vals = torch.cat([top_vals, first_val])
                    top_idxs = torch.cat([top_idxs, torch.tensor([0], device=self.device)])
                elif not is_pending:
                    # If you have to beat another flush, your best card needs to trump theirs!
                    valid_last_cards_in_suit = output_in_suit * valid_last_cards[suit::4]
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
            tensor = torch.zeros(52, device=self.device)
            tensor[best_option] = 1
            best_option = PlayCards(tensor, RoundType.HANDS, Hands.FLUSH)
        
        return best_option

    def find_best_full_house(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        if torch.sum(hand_type[:2]).item():
            is_pending = 1
        if not is_pending:
            cards_per_value = count_cards_per_value(prev_play)
            _, value_of_triple = torch.max(cards_per_value, dim=0)
            mask = torch.zeros(52, device=self.device)
            mask[value_of_triple*4: (value_of_triple + 1)*4] = 1
            prev_play = prev_play * mask

        best_triple = self.find_best_triple(output, card_list, prev_play, hand_type, is_pending, is_first_move)

        if best_triple is None:
            return None
        card_list = card_list * (1 - best_triple.cards)
        output[:52] = output[:52] * card_list

        # Edge case: the first triple is 3S 3H 3D, and it is the first move
        if is_first_move and torch.all(best_triple.cards[1:4]):
            best_triple.cards[3] = 0
            best_triple.cards[0] = 1
        # If it is the first move and you already included it in the triple, no need to include it in the pair!
        if best_triple.cards[0]:
            is_first_move = 0
        # Pair has is_pending enabled, because you can do any pair
        best_pair = self.find_best_pair(output, card_list, prev_play, hand_type, 1, is_first_move)
        if best_pair is None:
            return None
        
        return PlayCards(best_triple.cards + best_pair.cards, RoundType.HANDS, Hands.FULL_HOUSE)
        
    def find_best_four_hand(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        if torch.sum(hand_type[:3]).item():
            is_pending = 1
        
        if not is_pending:
            cards_per_value = count_cards_per_value(prev_play)
            _, value_of_triple = torch.max(cards_per_value, dim=0)
            mask = torch.zeros(52, device=self.device)
            mask[value_of_triple*4: (value_of_triple + 1)*4] = 1
            prev_play = prev_play * mask

        best_four = self.find_best_four(output, card_list, prev_play, hand_type, is_pending, 0)

        if best_four is None:
            return None
            
        # If it is the first move and you already included it in the four, no need to include it in the single!
        if best_four.cards[0]:
            is_first_move = 0
        card_list = card_list * (1 - best_four.cards)
        output[:52] = output[:52] * card_list
        best_single = self.find_best_single(output, card_list, prev_play, hand_type, 1, is_first_move)
        if best_single is None:
            return None
        
        return PlayCards(best_four.cards + best_single.cards, RoundType.HANDS, Hands.FOUR_OF_A_KIND)

    def find_best_straight_flush(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        # If it is the first move, only possible straight flush is 3C 4C 5C 6C 7C
        if is_first_move:
            if torch.all(card_list[0:20:4]):
                cards = torch.zeros(52, device=self.device)
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
                mask = torch.zeros(52, device=self.device)
                mask[suit::4] = 1
                masked_card_list = card_list * mask
                masked_output = output[:52] * masked_card_list
                best_straight = self.find_best_straight(masked_output, masked_card_list, prev_play, hand_type, is_pending, is_first_move)

                if best_straight is not None:
                    score = torch.sum(output[:52][best_straight.cards.to(torch.bool)])
                    scores.append(score)
                    actions.append(best_straight)
        
        best_action = actions[self.selection_function(torch.Tensor(scores), 1)]
        if best_action is not None:
            best_action.hands = Hands.STRAIGHT_FLUSH
        return best_action

    def return_pass(self, output, card_list, prev_play, hand_type, is_pending, is_first_move):
        return Pass()
        
    def clear_instances(self):
        self.instances = []

class TrainingDecisionFunction(Neural):
    def selection_function(self: Neural, probs: torch.Tensor, num_samples: int):
        try:
            if self.debug:
                print(probs)
            return torch.multinomial(probs, num_samples)
        except RuntimeError as e:
            output = self.model.actor(self.input)
            cloned = output.clone()
            output[:52] = F.softmax(output[:52], dim=0)
            output[52:57] = F.softmax(output[52:57], dim=0)
            output[57:] = F.softmax(output[57:], dim=0)
            raise RuntimeError(f"""
            Error calling the selection function. Here are the probs: {probs}
            Here is the card list: {output[18:70]}
            Here are the unnormalized logits: {cloned}
            Here are the normalized logits plus eps: {output}
            Good luck!""")




            
    
            
            
        








        
    