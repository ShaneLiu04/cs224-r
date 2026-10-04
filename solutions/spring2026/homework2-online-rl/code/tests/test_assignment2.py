"""Unit and integration tests for Homework 2 implementations."""

import pathlib
import sys
import types

import numpy as np
import torch

sys.modules.setdefault("hydra", types.SimpleNamespace())
sys.modules.setdefault("omegaconf", types.SimpleNamespace(OmegaConf=object))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gridworld_q_learning import SCENARIOS, run_all_scenarios
from off_policy import ACAgent
from on_policy import PPOAgent


def repeat_batch(batch):
    """Yield the same synthetic replay batch indefinitely."""
    while True:
        yield batch


def make_batch(batch_size=32, obs_dim=6, action_dim=2):
    """Construct a finite-valued replay batch."""
    rng = np.random.default_rng(7)
    obs = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    action = np.tanh(obs[:, :action_dim]).astype(np.float32)
    reward = rng.normal(size=(batch_size, 1)).astype(np.float32)
    discount = np.full((batch_size, 1), 0.99, dtype=np.float32)
    next_obs = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    return obs, action, reward, discount, next_obs


def test_gridworld_scenarios_reach_expected_outcomes():
    results = run_all_scenarios()
    assert len(results) == len(SCENARIOS)
    for scenario, result in zip(SCENARIOS, results):
        assert result["observed_outcome"] == scenario.expected_outcome
        assert np.isfinite(result["total_reward"])
        assert len(result["actions"]) <= scenario.horizon


def test_off_policy_updates_are_finite_and_change_parameters():
    torch.manual_seed(3)
    agent = ACAgent(
        obs_shape=(6,),
        action_shape=(2,),
        device="cpu",
        lr=1e-3,
        hidden_dim=32,
        num_critics=4,
        critic_target_tau=0.05,
        stddev_clip=0.3,
    )
    batch = make_batch()

    critic_before = [
        parameter.detach().clone() for parameter in agent.critic.parameters()
    ]
    critic_metrics = agent.update_critic(repeat_batch(batch))
    assert all(np.isfinite(value) for value in critic_metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, agent.critic.parameters())
    )

    actor_before = [
        parameter.detach().clone() for parameter in agent.actor.parameters()
    ]
    actor_metrics = agent.update_actor(repeat_batch(batch))
    assert all(np.isfinite(value) for value in actor_metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters())
    )


def test_behavior_cloning_reduces_action_error():
    torch.manual_seed(5)
    agent = ACAgent(
        obs_shape=(6,),
        action_shape=(2,),
        device="cpu",
        lr=3e-3,
        hidden_dim=32,
        num_critics=2,
        critic_target_tau=0.01,
        stddev_clip=0.3,
    )
    batch = make_batch(batch_size=128)
    obs = torch.from_numpy(batch[0])
    action = torch.from_numpy(batch[1])
    with torch.no_grad():
        initial_error = torch.mean((agent.actor(obs).mean - action) ** 2)
    for _ in range(100):
        metrics = agent.bc(repeat_batch(batch))
    with torch.no_grad():
        final_error = torch.mean((agent.actor(obs).mean - action) ** 2)
    assert np.isfinite(metrics["pretrain_actor_loss"])
    assert final_error < initial_error


def test_ppo_update_is_finite_and_changes_policy():
    torch.manual_seed(9)
    agent = PPOAgent(
        obs_shape=(3,),
        action_shape=(2,),
        device="cpu",
        lr=1e-3,
        batch_size=8,
        hidden_dim=16,
        clip_eps=0.2,
        ppo_epochs=2,
        value_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        gamma=0.99,
        reverse_kl_coef=0.01,
    )
    agent.set_reference_policy()
    observations = torch.randn(32, 3)
    with torch.no_grad():
        distribution = agent.actor(observations)
        actions = distribution.sample()
        old_log_prob = distribution.log_prob(actions).sum(-1)
    rewards = torch.randn(32)
    discounts = torch.full((32,), 0.99)
    next_observations = torch.randn(32, 3)
    dones = torch.zeros(32)
    dones[7::8] = 1.0
    rollout = [(
        observations,
        actions,
        rewards,
        discounts,
        next_observations,
        dones,
        old_log_prob,
    )]
    actor_before = [
        parameter.detach().clone() for parameter in agent.actor.parameters()
    ]
    metrics = agent.update(rollout)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters())
    )
