"""Unit tests for the HW3 TODOs (AWAC / IQL). Run: python -m pytest tests -q"""
import numpy as np
import pytest
import torch
from torch import nn

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure.utils import (
    OptimizerSpec, create_boxenv_q_network, create_boxenv_v_network,
    make_continuous_q_network, make_value_network)

ptu.init_gpu(use_gpu=False)

from cs224r.critics.awac_critic import AWACCritic  # noqa: E402
from cs224r.critics.iql_critic import IQLCritic  # noqa: E402
from cs224r.policies.MLP_policy import MLPPolicyAWAC  # noqa: E402


def _opt_spec(lr=1e-3):
    return OptimizerSpec(torch.optim.Adam, dict(lr=lr), lambda epoch: 1.0)


def _hparams(discrete, ob_dim=3, ac_dim=2, expectile=0.9):
    if discrete:
        q_func, v_func = create_boxenv_q_network, create_boxenv_v_network
    else:
        q_func = make_continuous_q_network(32, 2)
        v_func = make_value_network(32, 2)
    return dict(ob_dim=ob_dim, ac_dim=ac_dim, discrete=discrete, grad_norm_clipping=1e9,
                gamma=0.9, q_func=q_func, v_func=v_func, iql_expectile=expectile)


def _batch(discrete, n=16, ob_dim=3, ac_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    ob = rng.normal(size=(n, ob_dim)).astype(np.float32)
    nob = rng.normal(size=(n, ob_dim)).astype(np.float32)
    if discrete:
        ac = rng.integers(0, ac_dim, size=n).astype(np.float32)
    else:
        ac = rng.uniform(-1, 1, size=(n, ac_dim)).astype(np.float32)
    rew = rng.normal(size=n).astype(np.float32)
    done = (rng.uniform(size=n) < 0.3).astype(np.float32)
    return ob, ac, nob, rew, done


# ---------------------------------------------------------------- expectile
@pytest.mark.parametrize("zeta", [0.1, 0.5, 0.9])
def test_expectile_loss_weights(zeta):
    critic = IQLCritic(_hparams(True, expectile=zeta), _opt_spec())
    diff = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    loss = critic.expectile_loss(diff)
    expected = torch.tensor([1 - zeta, 1 - zeta, zeta, zeta]) * diff ** 2
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("zeta", [0.2, 0.5, 0.9])
def test_expectile_minimizer_matches_closed_form(zeta):
    """argmin_m E[L_ζ(x-m)] solves ζE[(x-m)+] = (1-ζ)E[(m-x)+]; 0.5 gives the mean."""
    critic = IQLCritic(_hparams(True, expectile=zeta), _opt_spec())
    x = torch.tensor([0.0, 0.0, 0.0, 10.0])
    m = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([m], lr=0.05)
    for _ in range(3000):
        opt.zero_grad()
        critic.expectile_loss(x - m).mean().backward()
        opt.step()
    m_star = m.item()
    lhs = zeta * torch.clamp(x - m_star, min=0).sum()
    rhs = (1 - zeta) * torch.clamp(m_star - x, min=0).sum()
    assert abs(lhs.item() - rhs.item()) < 1e-2
    if zeta == 0.5:
        assert abs(m_star - 2.5) < 1e-2
    # larger ζ -> estimate moves toward the upper tail
    assert (m_star > 2.5) == (zeta > 0.5) or zeta == 0.5


# ---------------------------------------------------------------- AWAC critic
@pytest.mark.parametrize("discrete", [True, False])
def test_awac_critic_target_and_masking(discrete):
    torch.manual_seed(0)
    critic = AWACCritic(_hparams(discrete), _opt_spec(lr=0.0))
    ob, ac, nob, rew, done = _batch(discrete)
    if discrete:
        next_ac = torch.randint(0, 2, (len(ob),))
    else:
        next_ac = torch.rand(len(ob), 2) * 2 - 1
    with torch.no_grad():
        nq = critic.get_target_q(torch.from_numpy(nob), next_ac)
        y = torch.from_numpy(rew) + 0.9 * (1 - torch.from_numpy(done)) * nq
        acs = torch.from_numpy(ac).long() if discrete else torch.from_numpy(ac)
        q1 = critic._get_q_value(critic.q_net, torch.from_numpy(ob), acs)
        q2 = critic._get_q_value(critic.q_net2, torch.from_numpy(ob), acs)
    out = critic.update(ob, ac, nob, rew, done, next_ac)
    assert np.isclose(out['Training Loss'], ((q1 - y) ** 2).mean().item(), atol=1e-5)
    assert np.isclose(out['Training Loss2'], ((q2 - y) ** 2).mean().item(), atol=1e-5)


def test_awac_critic_does_not_update_targets():
    critic = AWACCritic(_hparams(False), _opt_spec(lr=1e-2))
    before = [p.clone() for p in critic.q_net_target.parameters()]
    ob, ac, nob, rew, done = _batch(False)
    critic.update(ob, ac, nob, rew, done, torch.zeros(len(ob), 2))
    for b, p in zip(before, critic.q_net_target.parameters()):
        assert torch.equal(b, p)
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in critic.q_net_target.parameters())


def test_awac_critic_fits_terminal_reward():
    """With done=1 everywhere the critic must regress onto r alone."""
    torch.manual_seed(0)
    critic = AWACCritic(_hparams(False), _opt_spec(lr=3e-3))
    ob, ac, nob, rew, _ = _batch(False, n=32)
    done = np.ones_like(rew)
    for _ in range(800):
        out = critic.update(ob, ac, nob, rew, done, torch.zeros(len(ob), 2))
    assert out['Training Loss'] < 1e-2 and out['Training Loss2'] < 1e-2


# ---------------------------------------------------------------- IQL critic
@pytest.mark.parametrize("discrete", [True, False])
def test_iql_q_target_uses_v_of_next_state(discrete):
    torch.manual_seed(0)
    critic = IQLCritic(_hparams(discrete), _opt_spec(lr=0.0))
    ob, ac, nob, rew, done = _batch(discrete)
    with torch.no_grad():
        nv = critic.v_net(torch.from_numpy(nob)).squeeze(-1)
        y = torch.from_numpy(rew) + 0.9 * (1 - torch.from_numpy(done)) * nv
        acs = torch.from_numpy(ac).long() if discrete else torch.from_numpy(ac)
        q1 = critic._get_q_value(critic.q_net, torch.from_numpy(ob), acs)
    out = critic.update_q(ob, ac, nob, rew, done)
    assert np.isclose(out['Training Q Loss'], ((q1 - y) ** 2).mean().item(), atol=1e-5)


def test_iql_v_update_only_touches_v_net():
    critic = IQLCritic(_hparams(False), _opt_spec(lr=1e-2))
    q_before = [p.clone() for p in critic.q_net.parameters()]
    v_before = [p.clone() for p in critic.v_net.parameters()]
    ob, ac, *_ = _batch(False)
    out = critic.update_v(ob, ac)
    assert np.isfinite(out['Training V Loss'])
    assert all(torch.equal(b, p) for b, p in zip(q_before, critic.q_net.parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(v_before, critic.v_net.parameters()))


def test_iql_v_tracks_upper_expectile_of_q():
    """V(s) fitted on a fixed Q should sit above mean_a Q for ζ=0.9 and below for ζ=0.1."""
    torch.manual_seed(0)
    results = {}
    for zeta in (0.1, 0.9):
        critic = IQLCritic(_hparams(True, ob_dim=2, ac_dim=4, expectile=zeta), _opt_spec(3e-3))
        ob = np.zeros((256, 2), dtype=np.float32)
        ac = np.tile(np.arange(4), 64).astype(np.float32)
        with torch.no_grad():
            for net in (critic.q_net_target, critic.q_net2_target):
                net[-1].weight.zero_()
                net[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0, 10.0]))
        for _ in range(1500):
            critic.update_v(ob, ac)
        results[zeta] = critic.v_net(torch.zeros(1, 2)).item()
    assert results[0.1] < 3.25 < results[0.9]


# ---------------------------------------------------------------- AWR actor
def test_awac_weights_exp_over_lambda_and_clipped():
    torch.manual_seed(0)
    pol = MLPPolicyAWAC(ac_dim=2, ob_dim=3, n_layers=2, size=16, discrete=False,
                        learning_rate=0.0, lambda_awac=0.5)
    ob = torch.randn(4, 3)
    ac = torch.rand(4, 2) * 2 - 1
    adv = torch.tensor([-1.0, 0.0, 0.5, 100.0])
    with torch.no_grad():
        logp = pol(ob).log_prob(ac)
    loss = pol.update(ob, ac, adv_n=adv)
    w = torch.exp(adv / 0.5).clamp(max=50.0)
    assert np.isclose(loss, -(logp * w).mean().item(), atol=1e-5)


def test_awr_prefers_high_advantage_actions():
    """Two dataset actions per state; the one with positive advantage should dominate."""
    torch.manual_seed(0)
    pol = MLPPolicyAWAC(ac_dim=3, ob_dim=2, n_layers=2, size=32, discrete=True,
                        learning_rate=3e-3, lambda_awac=0.2)
    ob = torch.zeros(64, 2)
    ac = torch.tensor([0, 2] * 32)
    adv = torch.where(ac == 2, torch.tensor(1.0), torch.tensor(-1.0))
    for _ in range(300):
        pol.update(ob, ac, adv_n=adv)
    probs = pol(torch.zeros(1, 2)).probs[0]
    assert probs[2] > 0.9 and probs[1] < 0.05


# ---------------------------------------------------------------- agents
def _agent_params(discrete):
    hp = _hparams(discrete)
    hp.update(batch_size=8, learning_freq=1, optimizer_spec=_opt_spec(), n_layers=2, size=16,
              learning_rate=1e-3, awac_lambda=0.3, rew_shift=0.0, rew_scale=1.0,
              replay_buffer_size=64, num_timesteps=10)
    return hp


def _fill_buffer(agent, discrete):
    ob, ac, nob, rew, done = _batch(discrete, n=40)
    buf = agent.replay_buffer
    buf.obs = np.zeros((buf.size, 3), dtype=np.float32)
    buf.action = np.zeros((buf.size,) if discrete else (buf.size, 2), dtype=np.float32)
    buf.reward = np.zeros(buf.size, dtype=np.float32)
    buf.done = np.zeros(buf.size, dtype=bool)
    buf.obs[:40], buf.action[:40], buf.reward[:40], buf.done[:40] = ob, ac, rew, done > 0
    buf.num_in_buffer = 40


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("algo", ["awac", "iql"])
def test_agent_train_step(algo, discrete):
    torch.manual_seed(0)
    if algo == "awac":
        from cs224r.agents.awac_agent import AWACAgent as Agent
    else:
        from cs224r.agents.iql_agent import IQLAgent as Agent
    agent = Agent(None, _agent_params(discrete))
    _fill_buffer(agent, discrete)
    batch = agent.sample(8)
    adv = agent.estimate_advantage(batch[0], batch[1])
    assert adv.shape == (8,) and not adv.requires_grad
    log = agent.train(*batch)
    assert np.isfinite(log['Actor Loss'])
    assert agent.num_param_updates == 1


def test_iql_advantage_equals_q_minus_v():
    from cs224r.agents.iql_agent import IQLAgent
    agent = IQLAgent(None, _agent_params(False))
    ob, ac, *_ = _batch(False, n=8)
    adv = agent.estimate_advantage(ob, ac)
    with torch.no_grad():
        o, a = torch.from_numpy(ob), torch.from_numpy(ac)
        ref = agent.critic.get_q(o, a) - agent.critic.v_net(o).squeeze(-1)
    assert torch.allclose(adv, ref)
