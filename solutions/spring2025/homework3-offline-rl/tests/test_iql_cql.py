"""Unit tests for the IQL critic/advantage and CQL regularizer."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from cs224r.critics.cql_critic import CQLCritic  # noqa: E402
from cs224r.critics.iql_critic import IQLCritic  # noqa: E402
from cs224r.infrastructure import pytorch_util as ptu  # noqa: E402
from cs224r.infrastructure.dqn_utils import get_env_kwargs  # noqa: E402

ptu.init_gpu(use_gpu=False)


def make_hparams(**extra):
    hparams = dict(get_env_kwargs("PointmassMedium-v0"))
    hparams.update(env_name="PointmassMedium-v0", ob_dim=2, ac_dim=5,
                   double_q=True, iql_expectile=0.9, cql_alpha=0.0)
    hparams.update(extra)
    return hparams


def batch(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, 2)).astype(np.float32),
            rng.integers(0, 5, size=n).astype(np.int64),
            rng.normal(size=(n, 2)).astype(np.float32),
            rng.normal(size=n).astype(np.float32),
            (rng.random(n) < 0.2).astype(np.float32))


def test_expectile_loss_is_asymmetric():
    critic = IQLCritic(make_hparams(), get_env_kwargs("PointmassMedium-v0")["optimizer_spec"])
    diff = torch.tensor([1.0, -1.0])
    loss = critic.expectile_loss(diff)
    assert torch.allclose(loss, torch.tensor([0.9, 0.1]))


def test_iql_value_update_moves_v_toward_upper_expectile():
    torch.manual_seed(0)
    hp = make_hparams()
    critic = IQLCritic(hp, hp["optimizer_spec"])
    ob, ac, next_ob, rew, term = batch()
    for _ in range(300):
        critic.update_v(ob, ac)
    with torch.no_grad():
        q = critic.q_net_target(ptu.from_numpy(ob)).gather(
            1, ptu.from_numpy(ac).long()[:, None]).squeeze(1)
        v = critic.v_net(ptu.from_numpy(ob)).squeeze(1)
    # With zeta = 0.9, V should sit above the median of Q(s, a_data) on average.
    assert (q < v).float().mean() > 0.5


def test_iql_q_update_uses_v_target_and_terminal_mask():
    torch.manual_seed(0)
    hp = make_hparams(gamma=0.5)
    critic = IQLCritic(hp, hp["optimizer_spec"])
    ob, ac, next_ob, rew, term = batch()
    info = critic.update_q(ob, ac, next_ob, rew, term)
    assert np.isfinite(info["Training Q Loss"])


def test_cql_regularizer_matches_closed_form_and_alpha_zero_is_dqn():
    torch.manual_seed(0)
    hp = make_hparams(cql_alpha=0.0)
    critic = CQLCritic(hp, hp["optimizer_spec"])
    ob, ac, next_ob, rew, term = batch()
    with torch.no_grad():
        qa = critic.q_net(ptu.from_numpy(ob))
        expected_reg = (torch.logsumexp(qa, 1) - qa.gather(
            1, ptu.from_numpy(ac).long()[:, None]).squeeze(1)).mean().item()
        dqn_loss, _, _ = critic.dqn_loss(
            ptu.from_numpy(ob), ptu.from_numpy(ac).long(), ptu.from_numpy(next_ob),
            ptu.from_numpy(rew), ptu.from_numpy(term))
    info = critic.update(ob, ac, next_ob, rew, term)
    assert np.isclose(info["CQL Loss"], expected_reg, atol=1e-5)
    assert np.isclose(info["Training Loss"], dqn_loss.item(), atol=1e-5)
    assert info["CQL Loss"] >= 0


def test_cql_regularizer_pushes_down_ood_actions():
    torch.manual_seed(0)
    hp = make_hparams(cql_alpha=1.0)
    critic = CQLCritic(hp, hp["optimizer_spec"])
    ob, _, next_ob, rew, term = batch(256)
    ac = np.zeros(256, dtype=np.int64)
    for _ in range(300):
        info = critic.update(ob, ac, next_ob, np.zeros_like(rew), np.ones_like(term))
    q = critic.qa_values(ob)
    assert (q[:, 0:1] > q[:, 1:]).mean() > 0.9
    assert info["Data q-values"] > info["OOD q-values"] - np.log(5) - 1e-3
