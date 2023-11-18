from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, unpack_sequence
from torch.masked import masked_tensor

from pusoy.utils import Hands, RoundType
from pusoy.action import Action


class PusoyModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def get_hidden_state_from_input(
        self,
        player_no: int,
        card_list: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
        prev_play: torch.Tensor,
        prev_player: int,
        played_cards: list[torch.Tensor],
        hx: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def get_card_list_embeddings(self, card_list: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_action_probabilities(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_card_probabilities(
        self, output: torch.Tensor, card_list_embeddings: torch.Tensor, round_idx: int
    ) -> torch.Tensor:
        raise NotImplementedError


class FullLSTMModel(PusoyModel):
    def __init__(
        self,
        card_embedding_size: int = 64,
        num_card_embedding_size: int = 4,
        player_embedding_size: int = 16,
        hidden_size: int = 256,
        round_type_embedding_size: int = 64,
        first_move_embedding_size: int = 4,
    ):
        super().__init__()
        self.card_embeddings = nn.Embedding(52, card_embedding_size)
        self.num_cards_embeddings = nn.Embedding(13, num_card_embedding_size)
        self.player_embeddings = nn.Embedding(5, player_embedding_size, padding_idx=4)
        self.round_type_embeddings = nn.Embedding(9, round_type_embedding_size)
        self.first_move_embeddings = nn.Embedding(2, first_move_embedding_size)

        lstm_input_size = (
            (6 * card_embedding_size)
            + (4 * num_card_embedding_size)
            + (2 * player_embedding_size)
            + round_type_embedding_size
        )

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size)

        self.output_to_action = nn.Linear(hidden_size, round_type_embedding_size)
        self.output_to_card = nn.Linear(hidden_size, card_embedding_size)
        self.output_to_value = nn.Linear(hidden_size, 1)
        self.action_to_mask = nn.Linear(round_type_embedding_size, card_embedding_size)

        self.device = torch.device("cpu")

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def process_set_cards(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Applies the process_set function to a one-hot vector of cards,
        returning a single tensor representation of a list of cards.

        Args:
        - cards: one-hot vector of a hand of cards.
        """
        card_idxs = torch.nonzero(cards).flatten().to(self.device)
        if len(card_idxs) == 0:
            return torch.zeros(self.card_embeddings.embedding_dim).to(self.device)
        card_embeddings = self.card_embeddings(card_idxs)
        pooled_embeddings = torch.max(card_embeddings, dim=-2)[0]
        return pooled_embeddings

    def get_input_vector(
        self,
        player_no: int,
        card_list: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
        prev_play: torch.Tensor,
        prev_player: int,
        played_cards: list[torch.Tensor],
    ) -> torch.Tensor:
        """Creates the input vector to pass into the LSTM."""

        card_list = card_list.to(self.device)

        player_card_embeddings = self.process_set_cards(card_list)
        prev_play_embeddings = self.process_set_cards(prev_play)
        prev_played_card_embeddings = torch.stack(
            [self.process_set_cards(cards) for cards in played_cards]
        ).flatten()

        num_cards = torch.IntTensor([len(torch.nonzero(t)) for t in played_cards]).to(
            self.device
        )
        num_cards_embeddings = self.num_cards_embeddings(num_cards).flatten()

        if prev_player is None:
            prev_player = 4
        try:
            player_embeddings = self.player_embeddings(
                torch.IntTensor([player_no, prev_player]).to(self.device)
            ).flatten()
        except RuntimeError as re:
            print(torch.IntTensor([player_no, prev_player]))
            raise re

        round_value = min(round_type.value, 3) + hand_type.value
        round_type_embedding = self.round_type_embeddings(
            torch.IntTensor([round_value]).to(self.device)
        ).flatten()

        input = torch.cat(
            (
                player_card_embeddings,
                prev_play_embeddings,
                prev_played_card_embeddings,
                num_cards_embeddings,
                player_embeddings,
                round_type_embedding,
            )
        )

        return input

    def get_hidden_state_from_input(
        self,
        player_no: int,
        card_list: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
        prev_play: torch.Tensor,
        prev_player: int,
        played_cards: list[torch.Tensor],
        hx: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        """
        Passes the following as inputs to the LSTM. Retrieves hidden state.
        """
        input = self.get_input_vector(
            player_no,
            card_list,
            round_type,
            hand_type,
            prev_play,
            prev_player,
            played_cards,
        ).reshape(1, -1)

        out, hx = self.lstm(input, hx)

        return out.flatten(), hx

    def get_card_list_embeddings(self, card_list: torch.Tensor) -> torch.Tensor:
        """
        Performs self-attention on the embeddings of a list of cards.

        Args:
        - card_list: one-hot vector of a list of cards
        """
        card_idxs = torch.nonzero(card_list).flatten().to(self.device)
        card_embeddings = self.card_embeddings(card_idxs)
        attn_scores = F.softmax(
            torch.matmul(card_embeddings, card_embeddings.transpose(-1, -2)),
            dim=1,
        )
        attn_vecs = torch.matmul(attn_scores, card_embeddings)
        return attn_vecs

    def get_action_probabilities(self, output: torch.Tensor) -> torch.Tensor:
        """
        Computes the probabilities of taking each type of action.

        Args:
        - output: produced by get_hidden_state_from_input

        Returns
        - probabilities: one probability for each of:
          [pass, single, double, triple, straight, flush, full house, four, straight flush]
        """
        action_vector = self.output_to_action(output)
        round_type_embeddings = self.round_type_embeddings(
            torch.arange(9).to(self.device)
        )
        round_logits = round_type_embeddings @ action_vector
        return F.softmax(round_logits, dim=0)

    def get_card_probabilities(
        self, output: torch.Tensor, card_list_embeddings: torch.Tensor, round_idx: int
    ) -> torch.Tensor:
        """
        Args:
        - output: shape (output_dim)
        - card_list_embeddings: shape (N x card_dim), N is the number of cards a player has

        Returns:
        - card_probabilities: shape (N)
        """
        card_vector = self.output_to_card(output)
        round_embedding = self.round_type_embeddings(
            torch.IntTensor([round_idx]).to(self.device)
        )
        round_mask = F.sigmoid(self.action_to_mask(round_embedding))
        masked_card_list_embeddings = card_list_embeddings * round_mask
        logits = masked_card_list_embeddings @ card_vector
        return F.softmax(logits, dim=0)
    

    def get_ppo_output_vector(
        self, list_inputs: list[dict], list_actions: list[Action]
    ) -> torch.Tensor:
        """
        Returns the output vector containing both action and card probabilities that can
        directly be used in a PPO loss function.

        Args:
        - list_inputs: The list of inputs that a model observed during the course of a game.
        - list_actions: The list of actions a model took during the course of a game.

        Returns:
        - action_probabilities - The action probabilities at each timestep.
        - card_probabilities - The corresponding card probabilities for each action taken.
        """
        input_vectors = torch.stack(
            [self.get_input_vector(**input) for input in list_inputs]
        )
        outs, _ = self.lstm(input_vectors, None)

        action_vector = self.output_to_action(outs)
        round_type_embeddings = self.round_type_embeddings(
            torch.arange(9).to(self.device)
        )
        round_logits = torch.matmul(
            action_vector, round_type_embeddings.transpose(0, 1)
        )
        action_probabilities = F.softmax(round_logits, dim=1)
        action_idxs = torch.LongTensor([
            min(action.type.value, 3) + action.hand.value
            for action in list_actions
        ]).to(self.device)

        action_mask = F.one_hot(action_idxs, 9)

        card_list_embeddings = [
            self.get_card_list_embeddings(input["card_list"]) for input in list_inputs
        ]
        card_vectors = self.output_to_card(outs)
        round_embeddings = self.round_type_embeddings(action_idxs)
        round_masks = F.sigmoid(self.action_to_mask(round_embeddings))
        masked_card_list_embeddings = [
            e * mask for e, mask in zip(card_list_embeddings, round_masks)
        ]
        card_logits = [
            e @ vector for e, vector in zip(masked_card_list_embeddings, card_vectors)
        ]
        card_probabiltiies_flat = [F.softmax(c, dim=0) for c in card_logits]
        card_idxs = [torch.nonzero(input["card_list"]).flatten() for input in list_inputs]
        card_probabilities = torch.zeros(len(card_idxs), 52).to(self.device)
        for i in range(len(card_probabilities)):
            card_probabilities[i][card_idxs[i]] = card_probabiltiies_flat[i]
        
        card_mask = torch.stack([action.cards for action in list_actions]).to(self.device)

        probabilities = torch.hstack([action_probabilities, card_probabilities])
        mask = torch.hstack([action_mask, card_mask])

        return {
            "probs": probabilities[mask.nonzero(as_tuple=True)],
            "mask": mask,
            "value": self.output_to_value(outs).flatten(),
        }


class A2CLSTM(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.actor_1 = nn.Linear(input_dim, hidden_dim)
        self.actor_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, output_dim)
        self.critic_1 = nn.Linear(input_dim, hidden_dim)
        self.critic_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, input, states=None, compute_critic=True, packed=False):
        # x, states = self.lstm(input, states)
        if isinstance(input, PackedSequence):
            input = torch.cat(unpack_sequence(input), dim=0)
        #     x = torch.cat(unpack_sequence(x), dim=0)
        # x = torch.cat([x, input], dim=-1)
        x = input

        actor_x = F.relu(self.actor_1(x))
        actor_x = torch.cat([actor_x, input], dim=-1)
        actor_x = F.relu(self.actor_2(actor_x))
        actor_out = self.actor_out(actor_x)

        critic_out = None
        if compute_critic:
            critic_x = F.relu(self.critic_1(x))
            critic_x = torch.cat([critic_x, input], dim=-1)
            critic_x = F.relu(self.critic_2(critic_x))
            critic_out = self.critic_out(critic_x)

        return actor_out, critic_out, states


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
