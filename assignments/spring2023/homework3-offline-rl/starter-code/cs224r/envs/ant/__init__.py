from gym.envs.registration import register

register(
    id='ant-cs224r-v0',
    entry_point='cs224r.envs.ant:AntEnv',
    max_episode_steps=1000,
)
from cs224r.envs.ant.ant import AntEnv
