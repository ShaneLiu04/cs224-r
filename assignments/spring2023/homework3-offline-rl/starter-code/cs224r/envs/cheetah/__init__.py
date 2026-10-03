from gym.envs.registration import register

register(
    id='cheetah-cs224r-v0',
    entry_point='cs224r.envs.cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)
from cs224r.envs.cheetah.cheetah import HalfCheetahEnv
