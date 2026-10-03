from gym.envs.registration import register

register(
    id='reacher-cs224r-v0',
    entry_point='cs224r.envs.reacher:Reacher7DOFEnv',
    max_episode_steps=500,
)
from cs224r.envs.reacher.reacher_env import Reacher7DOFEnv
