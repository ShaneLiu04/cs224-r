from collections import deque, OrderedDict
from typing import Any, NamedTuple

import gym
import dm_env
import mujoco_py
import numpy as np
from dm_env import StepType, specs



class MetaWorldEnv:
    
  def __init__(self, name="hammer-v2", action_repeat=2, size=(84, 84), duration = 50):
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
      
      self.size = size
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
      
      # self.set_viewer_params(render_params[name])
      self.set_viewer_params(render_params)

      
  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)
 
  @property
  def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(0, 255, (self.size[0], self.size[1], 3), dtype=np.uint8)
        return gym.spaces.Dict(spaces)
 
  def set_viewer_params(self, params):
      self.viewer.cam.elevation = params["elevation"]
      self.viewer.cam.azimuth = params["azimuth"]
      self.viewer.cam.distance = params["distance"]
      self.viewer.cam.lookat[:] = params["lookat"][:]

  def step(self, action):
    reward = 0.0
    for _ in range(self.action_repeat):
        state, rew, done, info = self._env.step(action)
        reward += rew
        if done:
            break
    reward = 1.0 * info['success']
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    obs = {'image':img}
    self._step += 1
    if self._step >= self.duration:
        done = True
    return obs, reward, done, info

  def reset(self):
    self._env.hand_init_pos = self.hand_init_pose + 0.03 * np.random.normal(size = 3)
    _ = self._env.reset()
    for i in range(10):
        state,_,_,_ = self._env.step(np.zeros(self.action_space.shape))
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    obs = {'image':img}
    self._step = 0
    return obs
  
  def render(self, mode ='rgb_array', width = 84, height = 84):
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(width, height, depth=False)
      img = img[::-1]
      return img


class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def observation_spec(self):

      return OrderedDict(
          [(self._obs_key,  dm_env.specs.Array(
              shape = self._env.observation_space[self._obs_key].shape,
              dtype = self._env.observation_space[self._obs_key].dtype,
              name = self._obs_key))])
  
    
  def action_spec(self):
      return dm_env.specs.BoundedArray(
          shape = self._env.action_space.shape,
          minimum = self._env.action_space.low,
          maximum = self._env.action_space.high,
          dtype = self._env.action_space.dtype,
          name = 'action')

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    # print(obs[self._obs_key])
    return dm_env._environment.TimeStep(
        step_type = StepType.LAST if done else StepType.MID,
        reward = reward,
        discount = 1.0,
        observation = OrderedDict([(self._obs_key, obs[self._obs_key])]))

  def reset(self):
    obs = self._env.reset()
    # print(obs[self._obs_key])
    return dm_env._environment.TimeStep(
        step_type = StepType.FIRST,
        reward = 0.0,
        discount = 1.0,
        observation = OrderedDict([(self._obs_key, obs[self._obs_key])]))  


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(frame_stack):
    pixels_key = 'image'
    env = MetaWorldEnv(size = (84, 84))
    env = GymWrapper(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
