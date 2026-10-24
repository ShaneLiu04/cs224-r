"""Unit tests for the HER relabeling and DREAM decoder/reward code."""

import os
import sys
import types

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "code")
sys.path.insert(0, os.path.join(ROOT, "goal_conditioned_rl"))
sys.path.insert(0, os.path.join(ROOT, "meta_rl"))

import bit_flip_env  # noqa: E402
import q_network  # noqa: E402
import run_episode  # noqa: E402
import trainer  # noqa: E402
from utils import HERType  # noqa: E402


class ListBuffer:
    def __init__(self):
        self.items = []

    def add(self, state, action, reward, next_state):
        self.items.append((state, action, reward, next_state))


def reward_fn(x, y):
    return 0.0 if np.array_equal(x, y) else -1.0


def make_episode(num_bits=4, steps=4, seed=0):
    np.random.seed(seed)
    env = bit_flip_env.BitFlipEnv(num_bits)
    state, goal = env.reset()
    episode = []
    for t in range(steps):
        next_state, reward, done, _ = env.step(t % num_bits)
        episode.append((state.copy(), t % num_bits, float(reward), next_state.copy(), goal.copy()))
        state = next_state
        if done:
            break
    return episode


def test_no_hindsight_adds_only_original():
    episode = make_episode()
    buf = ListBuffer()
    trainer.update_replay_buffer(buf, episode, HERType.NO_HINDSIGHT, reward_fn, 4)
    assert len(buf.items) == len(episode)


def test_final_relabels_with_last_achieved_state():
    episode = make_episode()
    buf = ListBuffer()
    trainer.update_replay_buffer(buf, episode, HERType.FINAL, reward_fn, 4)
    assert len(buf.items) == 2 * len(episode)
    final_goal = episode[-1][3]
    relabeled = buf.items[1::2]
    for (state, _, reward, next_state), (_, _, _, ns, _) in zip(relabeled, episode):
        n = len(final_goal)
        np.testing.assert_array_equal(state[n:], final_goal)
        np.testing.assert_array_equal(next_state[n:], final_goal)
        assert reward == reward_fn(ns, final_goal)
    assert relabeled[-1][2] == 0.0


@pytest.mark.parametrize("her_type", [HERType.FUTURE, HERType.RANDOM])
def test_future_and_random_counts_and_rewards(her_type):
    episode = make_episode(num_bits=6, steps=6, seed=3)
    buf = ListBuffer()
    k = 4
    trainer.update_replay_buffer(buf, episode, her_type, reward_fn, k)
    assert len(buf.items) == (1 + k) * len(episode)
    n = 6
    achieved = [step[3] for step in episode]
    for t in range(len(episode)):
        block = buf.items[t * (1 + k):(t + 1) * (1 + k)]
        for state, _, reward, next_state in block[1:]:
            goal = state[n:]
            idx = [j for j, a in enumerate(achieved) if np.array_equal(a, goal)]
            assert idx, "relabeled goal must be an achieved state from the episode"
            if her_type == HERType.FUTURE:
                assert max(idx) >= t
            assert reward == reward_fn(next_state[:n], goal)


def test_relabeled_goal_is_a_copy():
    episode = make_episode()
    buf = ListBuffer()
    trainer.update_replay_buffer(buf, episode, HERType.FINAL, reward_fn, 4)
    episode[-1][3][:] = 7.0
    assert not np.any(buf.items[1][0] == 7.0)


def test_run_episode_greedy_and_success_flag():
    torch.manual_seed(0)
    np.random.seed(0)
    env = bit_flip_env.BitFlipEnv(3)
    net = q_network.QNetwork(6, 3)
    episode, ret, succeeded = run_episode.run_episode(env, net, 3)
    assert 1 <= len(episode) <= 3
    assert ret == sum(step[2] for step in episode)
    state, action, _, _, goal = episode[0]
    with torch.no_grad():
        q = net(torch.from_numpy(np.concatenate([state, goal]).astype(np.float32))[None])
    assert action == int(q.argmax())
    assert succeeded == np.array_equal(episode[-1][3], episode[-1][4])


def _load_encoder_decoder():
    from embed.encoder_decoder import EncoderDecoder
    return EncoderDecoder


def test_dream_decoder_loss_is_masked_mse_with_stopgrad():
    EncoderDecoder = _load_encoder_decoder()
    torch.manual_seed(0)
    z = torch.randn(2, 3, requires_grad=True)
    g = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).bool()
    losses = EncoderDecoder._compute_losses(types.SimpleNamespace(), None, z, g, None, mask)
    per_step = ((g - z[:, None]) ** 2).sum(-1)
    expected = (per_step * mask).sum() / mask.sum()
    assert torch.allclose(losses["decoder_loss"], expected)
    losses["decoder_loss"].backward()
    assert z.grad is None or torch.all(z.grad == 0)
    assert g.grad is not None and g.grad.abs().sum() > 0


def test_dream_rewards_are_information_gain():
    EncoderDecoder = _load_encoder_decoder()
    z = torch.tensor([[0.0, 0.0]])
    g = torch.tensor([[[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones(1, 4).bool()
    stub = types.SimpleNamespace(
        _penalty=0.1,
        _compute_embeddings=lambda traj: (z, g, None, mask),
    )
    rewards, distances = EncoderDecoder.label_rewards(stub, None)
    assert torch.allclose(distances, torch.tensor([[4.0, 1.0, 0.0, 1.0]]))
    assert torch.allclose(rewards, torch.tensor([[3.0, 1.0, -1.0]]) - 0.1)
    assert not rewards.requires_grad
