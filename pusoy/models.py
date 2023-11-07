import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import unpack_sequence, PackedSequence

from pusoy.utils import RoundType, Hands


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class FullLSTMModel(nn.Module):
    def __init__(
        self,
        card_embedding_size: int = 64,
        num_card_embedding_size: int = 4,
        player_embedding_size: int = 16,
        hidden_size: int = 256,
        round_type_embedding_size: int = 64,
        hand_type_embedding_size: int = 16,
        first_move_embedding_size: int = 4,
    ):
        self.card_embeddings = nn.Embeddings(52, card_embedding_size)
        self.num_cards_embeddings = nn.Embeddings(13, num_card_embedding_size)
        self.player_embeddings = nn.Embeddings(4, player_embedding_size)
        self.round_type_embeddings = nn.Embeddings(5, round_type_embedding_size)
        self.hand_type_embeddings = nn.Embeddings(
            6, hand_type_embedding_size, padding_idx=0
        )
        self.first_move_embeddings = nn.Embeddings(2, first_move_embedding_size)

        lstm_input_size = (
            (6 * card_embedding_size)
            + (4 * num_card_embedding_size)
            + (2 * player_embedding_size)
            + round_type_embedding_size
            + hand_type_embedding_size
            + first_move_embedding_size
        )

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size)

    def process_set_cards(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Applies the process_set function to a one-hot vector of cards,
        returning a single tensor representation of a list of cards.

        Args:
        - cards: either of shape (S) or (B x S), one-hot vector/s of a hand of cards.
        """
        card_idxs = torch.nonzero(cards).flatten()
        card_embeddings = self.card_embeddings(card_idxs)
        pooled_embeddings = torch.max(card_embeddings, dim=-2)[0]
        return pooled_embeddings

    def get_hidden_state_from_input(
        self,
        player_no: int,
        card_list: torch.Tensor,
        round_type: RoundType,
        hand_type: Hands,
        prev_play: torch.Tensor,
        prev_player: int,
        played_cards: list[torch.Tensor],
        is_first_move: bool,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ):
        """
        Passes the following as inputs to the LSTM. Retrieves hidden state.
        """
        player_card_embeddings = self.process_set_cards(card_list)
        prev_play_embeddings = self.process_set_cards(prev_play)
        prev_played_card_embeddings = self.process_set_cards(
            torch.stack(played_cards)
        ).flatten()

        num_cards = [len(torch.nonzero(t)[0]) for t in played_cards]
        num_cards_embeddings = self.num_cards_embeddings(num_cards).flatten()

        curr_player_embedding = self.player_embeddings(player_no)
        prev_player_embedding = self.player_embeddings(prev_player)

        round_type_embedding = self.round_type_embeddings(min(round_type.value, 4))
        hand_type_embedding = self.hand_type_embeddings(hand_type.value)
        first_move_embedding = self.first_move_embeddings(int(is_first_move))

        input = torch.cat(
            (
                player_card_embeddings,
                prev_play_embeddings,
                prev_played_card_embeddings,
                num_cards_embeddings,
                curr_player_embedding,
                prev_player_embedding,
                round_type_embedding,
                hand_type_embedding,
                first_move_embedding,
            )
        )

        return self.lstm(input, (hidden_state, cell_state))


class A2CLSTM(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=330, output_dim=62):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        # self.actor_1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.actor_1 = nn.Linear(input_dim, hidden_dim)
        self.actor_2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, output_dim)
        # self.critic_1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
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
