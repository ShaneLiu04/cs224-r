from gym.envs.registration import register

register(
    id='obstacles-cs224r-v0',
    entry_point='cs224r.envs.obstacles:Obstacles',
    max_episode_steps=500,
)
from cs224r.envs.obstacles.obstacles_env import Obstacles
