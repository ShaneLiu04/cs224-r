import pathlib
import subprocess
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.modules.setdefault("hydra", types.SimpleNamespace())
sys.modules.setdefault("omegaconf", types.SimpleNamespace(OmegaConf=object))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from on_policy import PPOAgent


def make_agent(*, clip_eps=0.2, batch_size=8, ppo_epochs=1):
    return PPOAgent(
        obs_shape=(1,),
        action_shape=(1,),
        device="cpu",
        lr=1e-3,
        batch_size=batch_size,
        hidden_dim=8,
        clip_eps=clip_eps,
        ppo_epochs=ppo_epochs,
        value_coef=0.0,
        entropy_coef=0.0,
        gae_lambda=0.95,
        gamma=0.99,
        reverse_kl_coef=0.0,
    )


def test_compute_gae_matches_manual_recursion():
    agent = make_agent()
    rewards = torch.tensor([1.2, -0.3, 0.5, 2.0], dtype=torch.float32)
    values = torch.tensor([0.4, -0.1, 0.2, 0.8], dtype=torch.float32)
    next_values = torch.tensor([-0.1, 0.2, 0.8, -0.4], dtype=torch.float32)
    discounts = torch.tensor([0.99, 0.99, 0.99, 0.99], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    advantages, returns = agent.compute_gae(
        rewards, values, next_values, discounts, dones
    )
    expected_advantages = torch.tensor([0.5129, -0.2, 1.848162, 0.804], dtype=torch.float32)
    expected_returns = torch.tensor([0.9129, -0.3, 2.048162, 1.604], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages, atol=1e-6)
    assert torch.allclose(returns, expected_returns, atol=1e-6)


def test_clipped_surrogate_objective():
    new_log_prob = torch.log(torch.tensor([1.5, 0.7, 1.1], dtype=torch.float32))
    olp_ep = torch.zeros(3, dtype=torch.float32)
    adv_ep = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float32)
    clip_eps = 0.2

    ### Paste solution code for clipped surrogate (PPO-Clip objective)
    ### The exact solution code should suffice provided self.clip_eps is changed to clip_eps


    ###


    expected_policy_loss = torch.tensor(-0.26666668, dtype=torch.float32)

    assert torch.allclose(policy_loss, expected_policy_loss, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "pytest", __file__])
    )
