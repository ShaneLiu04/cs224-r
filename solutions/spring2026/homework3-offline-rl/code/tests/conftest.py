import importlib.util
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The unit tests only exercise torch code; stub gym when it is not installed (e.g. laptops).
if importlib.util.find_spec("gym") is None:
    registration = types.ModuleType("gym.envs.registration")
    registration.register = lambda **kwargs: None
    registration.registry = types.SimpleNamespace(env_specs={})
    envs = types.ModuleType("gym.envs")
    envs.registration = registration
    gym = types.ModuleType("gym")
    gym.envs = envs
    sys.modules.update({"gym": gym, "gym.envs": envs, "gym.envs.registration": registration})
