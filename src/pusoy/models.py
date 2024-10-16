from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from pusoy.constants import OUTPUT_SIZES
from pusoy.utils import Hands, RoundType


class PusoyModel(nn.Module, ABC):
    """Defines the model interface for Pusoy models.

    Models output raw logits. All conversion to probabilities should be done
    in the upstream applications.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def v(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pylint-disable
        pass


class DenseA2C(PusoyModel):
    """
    A dense model. Generates a common feature vector that is used as the
    baseline for both actor and critic models.
    """

    def __init__(self, input_size=330, hidden_size=256, output_size=62):
        super().__init__()
        self.preprocess = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
        )

        self.actor_1 = nn.Linear(hidden_size, hidden_size)
        self.actor_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.actor_out = nn.Linear(hidden_size, output_size)

        self.critic_1 = nn.Linear(hidden_size, hidden_size)
        self.critic_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Generates an action vector from an input state.

        Input:
            state - (batch_size, state_dim) A tensor representing state spaces

        Output:
            A vector of logits over the action space.
        """
        x = self.preprocess(state)

        actor_x = F.relu(self.actor_1(x))
        actor_x = torch.cat([actor_x, x], dim=-1)
        actor_x = F.relu(self.actor_2(actor_x))
        actor_out = self.actor_out(actor_x)
        actor_out = torch.split(actor_out, OUTPUT_SIZES, dim=-1)

        return actor_out

    def v(self, state: torch.Tensor) -> torch.Tensor:
        """Generates a state evalution from an input state.

        Input:
            state - (batch_size, state_dim) A tensor representing state spaces

        Output:
            A tensor containing the evaluations over the states.
        """
        x = self.preprocess(state)

        critic_x = F.relu(self.critic_1(x))
        critic_x = torch.cat([critic_x, x], dim=-1)
        critic_x = F.relu(self.critic_2(critic_x))
        critic_out = self.critic_out(critic_x)
        critic_out = critic_out.flatten()

        return critic_out

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates an action vector and a state evalution from an input state.

        Input:
            state - (batch_size, state_dim) A tensor representing state spaces

        Output:
            A tuple containing a tensor representing action logits
            for the different spaces, and a tensor evaluation over the states
            (batch_size,).
        """
        x = self.preprocess(state)

        actor_x = F.relu(self.actor_1(x))
        actor_x = torch.cat([actor_x, x], dim=-1)
        actor_x = F.relu(self.actor_2(actor_x))
        actor_out = self.actor_out(actor_x)

        critic_x = F.relu(self.critic_1(x))
        critic_x = torch.cat([critic_x, x], dim=-1)
        critic_x = F.relu(self.critic_2(critic_x))
        critic_out = self.critic_out(critic_x)
        critic_out = critic_out.flatten()

        return actor_out, critic_out


def create_input_tensor(
    card_list: torch.Tensor,
    prev_play: torch.Tensor | None,
    played_cards: list[torch.Tensor],
    round_type: RoundType,
    hand_type: Hands,
    player_no: int,
    prev_player: int,
) -> torch.Tensor:
    """Creates the input to the model based on the provided information.
    
    Args:
        card_list: List of cards that the player has.
        prev_play: Cards that were played the round prior.
        played_cards: A list of tensors representing all of the cards that have been played 
            thus far by each player.
        round_type: Current round type.
        hand_type: Current hand type, if any.
        player_no: The position of the current player.
        prev_player: The position of the player who played the previous move.
    
    Returns:
        An (input_dim) tensor representation of the input state.
    """

    if prev_play is None:
        prev_play = torch.zeros(52)
    player_no_vec, prev_player_vec = torch.zeros(4), torch.zeros(4)
    player_no_vec[player_no] = 1
    if prev_player is not None:
        prev_player_vec[prev_player] = 1

    round_type = round_type.to_tensor(dtype=torch.float)
    hand_type = hand_type.to_tensor(dtype=torch.float)

    return torch.cat(
        [round_type, hand_type, player_no_vec, prev_player_vec, card_list, prev_play]
        + played_cards
    )


def get_probs_from_logits(
    logits: torch.Tensor,
    card_list: torch.Tensor | None = None,
    round_type: RoundType | None = None,
    hand_type: Hands | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the selection probabilities based on the provided logits.
    Optionally performs masking if a round type or hand type is provided.
    
    Args:
        logits: (output_dim) tensor from the model
        card_list: list of cards available to the player
        round_type: current round type
        hand_type: current hand type
    
    Returns:
        A tuple of tensors representing card probs, round probs, and hand probs.
    """
    logits = logits.cpu()
    
    card_logits = logits[:52]
    action_logits = logits[52:57]
    hand_logits = logits[57:]

    if card_list:
        card_logits = card_logits - (1e16 * (1 - card_list))
    
    if round_type:
        action_mask = round_type.to_tensor(dtype=torch.bool)
        action_mask[1:] ^= action_mask[0]
        action_mask[0] ^= True
        action_logits = action_logits - (1e16 * (1 - action_mask))
    
    if hand_type:
        min_hand = max(hand_type.value - 1, 0)
        hand_logits[:min_hand] -= 1e16
    
    card_probs = F.softmax(card_logits, dim=0)
    action_probs = F.softmax(action_logits, dim=0)
    hand_probs = F.softmax(hand_logits, dim=0)

    return card_probs, action_probs, hand_probs
