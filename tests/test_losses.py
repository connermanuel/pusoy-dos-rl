import pytest
import torch

from pusoy.action import Pass, PlayCards
from pusoy.losses import entropy, gae, generate_action_masks, ppo_loss
from pusoy.utils import Hands, RoundType, indexes_to_one_hot


class TestGAE:
    state_values = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float32)

    def test_gae_lambda_zero(self):
        rewards = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
        next_state_values = torch.tensor([0.6, 0.7, 0.0], dtype=torch.float32)
        td_errors = rewards + 0.99 * next_state_values - self.state_values
        expected_advantages = td_errors

        advantages = gae(self.state_values, rewards, gamma=0.99, lambd=0.0)

        assert torch.allclose(
            advantages, expected_advantages, atol=1e-2
        ), f"Expected {expected_advantages}, but got {advantages}"

    def test_gae_lambda_one(self):
        rewards = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
        gamma = 0.99
        lambd = 1.0

        # Calculate expected advantages manually
        deltas = (
            rewards
            + gamma * torch.cat((self.state_values[1:], torch.tensor([0.0])))
            - self.state_values
        )
        expected_advantages = torch.zeros_like(deltas)
        gae_advantage = 0.0
        for t in reversed(range(len(deltas))):
            gae_advantage = deltas[t] + gamma * lambd * gae_advantage
            expected_advantages[t] = gae_advantage

        advantages = gae(self.state_values, rewards, gamma=gamma, lambd=lambd)

        assert torch.allclose(
            advantages, expected_advantages, atol=1e-2
        ), f"Expected {expected_advantages}, but got {advantages}"


class TestGenerateActionMasks:
    def test_generate_action_masks_basic(self):
        actions = [
            Pass(),
            PlayCards(indexes_to_one_hot(52, [0]), RoundType.SINGLES, Hands.NONE),
            PlayCards(indexes_to_one_hot(52, [0, 1]), RoundType.PAIRS, Hands.NONE),
            PlayCards(
                indexes_to_one_hot(52, [0, 4, 8, 12, 16]),
                RoundType.HANDS,
                Hands.STRAIGHT,
            ),
            PlayCards(
                indexes_to_one_hot(52, [0, 1, 4, 5, 6]),
                RoundType.HANDS,
                Hands.FULL_HOUSE,
            ),
        ]

        # Generate masks
        card_masks, round_masks, hand_masks = generate_action_masks(actions)

        # Check shapes
        assert card_masks.shape == (5, 52)
        assert round_masks.shape == (5, 5)
        assert hand_masks.shape == (5, 5)

        assert torch.all(card_masks[0] == 0)

        assert torch.all(round_masks[1] == RoundType.SINGLES.to_tensor())
        assert torch.all(round_masks[2] == RoundType.PAIRS.to_tensor())
        assert torch.all(round_masks[3] == RoundType.HANDS.to_tensor())
        assert torch.all(round_masks[4] == RoundType.HANDS.to_tensor())

        assert torch.all(hand_masks[3] == Hands.STRAIGHT.to_tensor())
        assert torch.all(hand_masks[4] == Hands.FULL_HOUSE.to_tensor())


class TestEntropy:
    def test_entropy_deterministic(self):
        logits = torch.tensor([[100.0, -100.0, -100.0]])
        log_probs = torch.log_softmax(logits, dim=-1)
        ent = entropy(log_probs)
        assert torch.allclose(ent, torch.tensor([0.0]), atol=1e-5)

    def test_entropy_uniform(self):
        n_actions = 3
        log_probs = torch.log(torch.ones(1, n_actions) / n_actions)
        ent = entropy(log_probs)
        expected = torch.tensor([torch.log(torch.tensor(n_actions).float())])
        assert torch.allclose(ent, expected, atol=1e-5)

    def test_entropy_batched(self):
        log_probs = torch.log_softmax(torch.randn(4, 5), dim=-1)
        ent = entropy(log_probs)
        assert ent.shape == (4,)
        assert torch.all(ent >= 0)

    def test_entropy_numerical_stability(self):
        log_probs = torch.tensor([[-1000.0, 0.0]])
        log_probs = torch.log_softmax(log_probs, dim=-1)
        ent = entropy(log_probs)
        assert not torch.isnan(ent).any()


class TestPPOLoss:
    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def setup_inputs(self, batch_size):
        # Create log probabilities for each action part
        curr_cards = torch.log_softmax(torch.randn(batch_size, 52), dim=-1)
        curr_hand = torch.log_softmax(torch.randn(batch_size, 5), dim=-1)
        curr_round = torch.log_softmax(torch.randn(batch_size, 5), dim=-1)

        prev_cards = curr_cards.detach().clone()
        prev_hand = curr_hand.detach().clone()
        prev_round = curr_round.detach().clone()

        # Create masks
        masks = (
            torch.ones(batch_size, 52),
            torch.ones(batch_size, 5),
            torch.ones(batch_size, 5),
        )

        return (
            (curr_cards, curr_hand, curr_round),
            (prev_cards, prev_hand, prev_round),
            torch.randn(batch_size),  # state values
            torch.randn(batch_size),  # advantages
            masks,
        )

    def test_zero_advantages(self, setup_inputs):
        curr_probs, prev_probs, values, _, masks = setup_inputs
        advantages = torch.zeros_like(values)

        loss, metrics = ppo_loss(curr_probs, prev_probs, values, advantages, masks)

        assert torch.allclose(metrics["actor_loss"], torch.tensor(0.0), atol=1e-5)

    def test_clipping(self, setup_inputs):
        curr_probs, prev_probs, values, advantages, masks = setup_inputs

        # Make current policy very different from old policy
        curr_probs = tuple(p + 100.0 for p in curr_probs)

        loss1, _ = ppo_loss(
            curr_probs, prev_probs, values, advantages, masks, eps_clip=0.2
        )
        loss2, _ = ppo_loss(
            curr_probs, prev_probs, values, advantages, masks, eps_clip=0.1
        )

        # Tighter clipping should give smaller loss
        assert loss2 < loss1

    def test_entropy_bonus(self, setup_inputs):
        curr_probs, prev_probs, values, advantages, masks = setup_inputs

        loss1, _ = ppo_loss(
            curr_probs, prev_probs, values, advantages, masks, c_entropy=0.01
        )
        loss2, _ = ppo_loss(
            curr_probs, prev_probs, values, advantages, masks, c_entropy=0.1
        )

        # Larger entropy coefficient should give different loss
        assert loss1 != loss2

    def test_advantage_scaling(self, setup_inputs):
        curr_probs, prev_probs, values, advantages, masks = setup_inputs

        loss1, _ = ppo_loss(curr_probs, prev_probs, values, advantages, masks)
        loss2, _ = ppo_loss(curr_probs, prev_probs, values, advantages * 2, masks)

        # Scaling advantages should scale actor loss
        assert abs(loss2) > abs(loss1)

    def test_identical_policies(self, setup_inputs):
        curr_probs, prev_probs, values, advantages, masks = setup_inputs

        loss, metrics = ppo_loss(curr_probs, curr_probs, values, advantages, masks)

        # KL should be zero for identical policies
        assert torch.allclose(
            torch.tensor(metrics["approx_kl"]), torch.tensor(0.0), atol=1e-5
        )
