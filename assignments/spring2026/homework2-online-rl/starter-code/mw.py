from collections import deque, OrderedDict
from typing import Any, NamedTuple

import gym
import dm_env
import mujoco_py
import numpy as np
from dm_env import StepType, specs



class MetaWorldEnv:
  """Wrap a Meta-World task with fixed reset, sparse rewards, and action repeat."""

  def __init__(self, name="hammer-v2", action_repeat=2, duration = 50):
      """Create the underlying Meta-World environment and configure rollout limits."""
      from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
      render_params={"elevation": -22.5,
                     "azimuth": 15,
                     "distance": 0.75,
                     "lookat": np.array([-0.15, 0.60, 0.25])}
      
      self._env = ALL_V2_ENVIRONMENTS[name]()
      self._env.max_path_length = np.inf
      self._env._freeze_rand_vec = False
      self._env._partially_observable = False
      self._env._set_task_called = True

      self.hand_init_pose = self._env.hand_init_pos.copy()
      self.hand_init_pose = np.array([0.1 , 0.5, 0.30])
      
      self.action_repeat = action_repeat
      self.duration = duration
      self._step = None

      
  def __getattr__(self, attr):
     """Forward unknown attributes to the wrapped Meta-World environment."""
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)
 
  @property
  def observation_space(self):
        """Expose the wrapped Gym observation space."""
        return self._env.observation_space
 
  def step(self, action):
    """Repeat one action, collapse the reward to success, and enforce max duration."""
    reward = 0.0
    for _ in range(self.action_repeat):
        state, rew, done, info = self._env.step(action)
        state = state.astype(self._env.observation_space.dtype)
        reward += rew
        if done:
            break
    reward = 1.0 * info['success']
    self._step += 1
    if self._step >= self.duration:
        done = True
    return state, reward, done, info

  def reset(self):
    """Randomize the hand start pose and return the warmed-up initial observation."""
    self._env.hand_init_pos = self.hand_init_pose + 0.03 * np.random.normal(size = 3)
    _ = self._env.reset()
    for i in range(10):
        state,_,_,_ = self._env.step(np.zeros(self.action_space.shape))
        state = state.astype(self._env.observation_space.dtype)
    self._step = 0
    return state
  
  def render(self, mode ='rgb_array', width = 84, height = 84):
      """Stub render method kept for interface compatibility."""
      return
      


class GymWrapper:
  """Adapt a Gym-style environment to the dm_env API expected by training code."""

  def __init__(self, env, act_key='action'):
    """Store the wrapped environment and action field name."""
    self._env = env
    self._act_key = act_key

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped environment."""
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def observation_spec(self):
      """Describe observations with a dm_env Array spec."""
      return dm_env.specs.Array(
              shape = self._env.observation_space.shape,
              dtype = self._env.observation_space.dtype,
              name = 'observation')
  
    
  def action_spec(self):
      """Describe actions with a dm_env bounded array spec."""
      return dm_env.specs.BoundedArray(
          shape = self._env.action_space.shape,
          minimum = self._env.action_space.low,
          maximum = self._env.action_space.high,
          dtype = self._env.action_space.dtype,
          name = 'action')

  def step(self, action):
    """Step the environment and convert the result into a dm_env TimeStep."""
    obs, reward, done, info = self._env.step(action)
    return dm_env._environment.TimeStep(
        step_type = StepType.LAST if done else StepType.MID,
        reward = reward,
        discount = 1.0,
        observation = obs)

  def reset(self):
    """Reset the environment and return a FIRST dm_env TimeStep."""
    obs = self._env.reset()
    return dm_env._environment.TimeStep(
        step_type = StepType.FIRST,
        reward = 0.0,
        discount = 1.0,
        observation = obs)  


class ExtendedTimeStep(NamedTuple):
    """TimeStep variant that always carries the action that produced it."""
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        """Return whether this timestep is the first step of an episode."""
        return self.step_type == StepType.FIRST

    def mid(self):
        """Return whether this timestep is an intermediate step."""
        return self.step_type == StepType.MID

    def last(self):
        """Return whether this timestep terminates the episode."""
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        """Support both tuple-style indexing and attribute-style lookup by name."""
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        """Cast the public action spec to a requested dtype while preserving bounds."""
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        """Cast the action back to the wrapped dtype before stepping the env."""
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        """Expose the wrapped observation spec unchanged."""
        return self._env.observation_spec()

    def action_spec(self):
        """Expose the rewritten action spec."""
        return self._action_spec

    def reset(self):
        """Reset the wrapped environment."""
        return self._env.reset()

    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped environment."""
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        """Wrap a dm_env environment so every timestep also includes an action."""
        self._env = env

    def reset(self):
        """Reset the env and synthesize a zero action for the initial timestep."""
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        """Step the env and attach the action to the returned timestep."""
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        """Convert a base timestep into an ExtendedTimeStep with defaulted fields."""
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        """Expose the wrapped observation spec."""
        return self._env.observation_spec()

    def action_spec(self):
        """Expose the wrapped action spec."""
        return self._env.action_spec()

    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped environment."""
        return getattr(self._env, name)


def make():
    env = MetaWorldEnv()
    env = GymWrapper(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = ExtendedTimeStepWrapper(env)
    return env
