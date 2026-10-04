"""Assignment 1 implementation tests."""

import numpy as np
import torch

from dagger import DeterministicExpert, rollout_and_relabel
from expert import COMMIT_DIST
from flappy_bird_env import FlappyBirdEnv
from losses import flow_matching_loss, mse_loss
from networks import BCPolicy, FlowMatchingPolicy, FlowMatchingSchedule


def test_bc_policy_shape_range_and_gradients():
    states = torch.rand(16, 4)
    actions = torch.rand(16, 20)
    policy = BCPolicy()
    predictions = policy(states)
    assert predictions.shape == actions.shape
    assert torch.all((predictions >= 0.0) & (predictions <= 1.0))
    loss = mse_loss(policy, states, actions)
    loss.backward()
    assert all(parameter.grad is not None for parameter in policy.parameters())


def test_flow_interpolation_endpoints_and_velocity():
    schedule = FlowMatchingSchedule(action_dim=20)
    actions = torch.rand(8, 20)
    torch.manual_seed(7)
    at_zero, velocity_zero = schedule.interpolate(actions, torch.zeros(8))
    torch.manual_seed(7)
    at_one, velocity_one = schedule.interpolate(actions, torch.ones(8))
    assert torch.allclose(at_one, actions)
    assert torch.allclose(velocity_zero, actions - at_zero)
    assert torch.allclose(velocity_one, actions - at_zero)


def test_flow_loss_and_sampling_are_finite():
    states = torch.rand(4, 4)
    actions = torch.rand(4, 20)
    policy = FlowMatchingPolicy(num_steps=4)
    loss = flow_matching_loss(policy, states, actions)
    loss.backward()
    samples = policy.schedule.sample(policy.model, states)
    assert torch.isfinite(loss)
    assert samples.shape == actions.shape
    assert torch.all((samples >= 0.0) & (samples <= 1.0))


def test_deterministic_expert_chooses_upper_gap():
    expert = DeterministicExpert(smoothing=1.0)
    observation = np.array(
        [COMMIT_DIST / 2, 0.25, 0.75, 0.5], dtype=np.float32,
    )
    assert np.isclose(expert.act(observation), observation[1])


def test_dagger_relabel_shapes():
    torch.manual_seed(1)
    policy = BCPolicy()
    states, actions = rollout_and_relabel(
        policy=policy,
        difficulty="hard",
        num_episodes=2,
        pipe_speed=3.0,
        seed=11,
        action_chunk=20,
        device=torch.device("cpu"),
    )
    assert states.ndim == 2 and states.shape[1] == 4
    assert actions.ndim == 2 and actions.shape[1] == 20
    assert len(states) == len(actions)
    assert np.isfinite(states).all() and np.isfinite(actions).all()


def test_environment_is_seed_reproducible():
    first = FlappyBirdEnv(difficulty="hard")
    second = FlappyBirdEnv(difficulty="hard")
    obs_a, _ = first.reset(seed=42)
    obs_b, _ = second.reset(seed=42)
    assert np.allclose(obs_a, obs_b)
    first.close()
    second.close()
