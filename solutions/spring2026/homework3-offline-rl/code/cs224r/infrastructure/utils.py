import copy
import random
from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim
import time
from gym.envs.registration import register, registry
from torch import nn


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array'),
                      deterministic=False):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=256, width=400)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=256, width=400)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        try:
            ac = policy.get_action(ob, deterministic=deterministic)
        except TypeError:
            ac = policy.get_action(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        if done or steps >= max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False,
                        render_mode=('rgb_array'), deterministic=False):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode,
                                 deterministic=deterministic)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')
    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False,
                          render_mode=('rgb_array'), deterministic=False):
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode,
                                 deterministic=deterministic)
        paths.append(path)
    return paths

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """Package separate rollout arrays into a single dict."""
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """Concatenate a list of rollout dicts into separate flat arrays."""
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):
    data = copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis=0)
    mean_data[mean_data == 0] = 0.000001  # avoid division by zero
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data


# ---------------------------------------------------------------------------
# Optimizer spec
# ---------------------------------------------------------------------------

OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)


# ---------------------------------------------------------------------------
# Environment registration
# ---------------------------------------------------------------------------

def register_custom_envs():
    if 'PointmassEasy-v0' not in registry.env_specs:
        register(
            id='PointmassEasy-v0',
            entry_point='cs224r.envs.pointmass.pointmass:Pointmass',
            kwargs={'difficulty': 0}
        )
    if 'PointmassMedium-v0' not in registry.env_specs:
        register(
            id='PointmassMedium-v0',
            entry_point='cs224r.envs.pointmass.pointmass:Pointmass',
            kwargs={'difficulty': 1}
        )
    if 'PointmassHard-v0' not in registry.env_specs:
        register(
            id='PointmassHard-v0',
            entry_point='cs224r.envs.pointmass.pointmass:Pointmass',
            kwargs={'difficulty': 2}
        )
    if 'PointmassVeryHard-v0' not in registry.env_specs:
        register(
            id='PointmassVeryHard-v0',
            entry_point='cs224r.envs.pointmass.pointmass:Pointmass',
            kwargs={'difficulty': 3}
        )


def get_env_kwargs(env_name):
    if 'Pointmass' in env_name:
        def pointmass_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': pointmass_optimizer(),
            'q_func': create_boxenv_q_network,
            'v_func': create_boxenv_v_network,
            'replay_buffer_size': int(1e5),
            'gamma': 0.95,
            'learning_freq': 1,
            'frame_history_len': 1,
            'grad_norm_clipping': 10,
            'num_timesteps': 50000,
            'env_wrappers': pointmass_empty_wrapper
        }

    elif 'antmaze' in env_name:
        def d4rl_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': d4rl_optimizer(lr=3e-4),
            'q_func': make_continuous_q_network(hidden_size=256, n_layers=2),
            'v_func': make_value_network(hidden_size=256, n_layers=2),
            'replay_buffer_size': int(2e6),
            'gamma': 0.99,
            'learning_freq': 1,
            'frame_history_len': 1,
            'grad_norm_clipping': 10,
            'num_timesteps': int(1e6),
            'env_wrappers': d4rl_empty_wrapper,
            'continuous': True,
        }

    else:
        raise NotImplementedError

    return kwargs


# ---------------------------------------------------------------------------
# Network builders
# ---------------------------------------------------------------------------

def create_boxenv_q_network(ob_dim, num_actions):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )

def create_boxenv_v_network(ob_dim):
    return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

class ContinuousQNet(nn.Module):
    """Q(s, a) network for continuous action spaces.
    Takes (obs, action) as separate tensors and returns a scalar Q-value per sample.
    """
    def __init__(self, ob_dim, ac_dim, hidden_size=256, n_layers=2):
        super().__init__()
        layers = []
        in_size = ob_dim + ac_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU()]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)  # [B, 1]


def make_continuous_q_network(hidden_size=256, n_layers=2):
    """Returns a factory `(ob_dim, ac_dim) -> ContinuousQNet`."""
    def factory(ob_dim, ac_dim):
        return ContinuousQNet(ob_dim, ac_dim, hidden_size, n_layers)
    return factory


def make_value_network(hidden_size=256, n_layers=2):
    """Returns a factory `(ob_dim) -> nn.Sequential` mapping obs -> scalar."""
    def factory(ob_dim):
        layers = []
        in_size = ob_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU()]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)
    return factory


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def d4rl_optimizer(lr=3e-4):
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(lr=lr),
        learning_rate_schedule=lambda epoch: 1.0,  # constant LR
    )

def pointmass_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(lr=1),
        learning_rate_schedule=lambda epoch: 1e-3,
    )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class ConstantSchedule(object):
    def __init__(self, value):
        self._v = value

    def value(self, t):
        return self._v


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)
        assert self._outside_value is not None
        return self._outside_value


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def sample_n_unique(sampling_f, n):
    """Sample n unique values from sampling_f."""
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class MemoryOptimizedReplayBuffer(object):
    def __init__(self, size, frame_history_len, float_obs=False,
                 continuous_action=False, ac_dim=1):
        self.float_obs = float_obs
        self.continuous_action = continuous_action
        self.ac_dim = ac_dim

        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs               = None
        self.action            = None
        self.reward            = None
        self.done              = None
        self.next_obs_explicit = None  # set by offline D4RL loader; overrides obs[idx+1]

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        if self.next_obs_explicit is not None:
            next_obs_batch = self.next_obs_explicit[idxes]
        else:
            next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx = idx + 1
        start_idx = end_idx - self.frame_history_len
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.float_obs else np.uint8)
            if self.continuous_action:
                ac_shape = [self.size, self.ac_dim] if self.ac_dim > 1 else [self.size]
                self.action = np.empty(ac_shape, dtype=np.float32)
            else:
                self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done   = np.empty([self.size], dtype=bool)
        self.obs[self.next_idx] = frame
        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        return ret

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
