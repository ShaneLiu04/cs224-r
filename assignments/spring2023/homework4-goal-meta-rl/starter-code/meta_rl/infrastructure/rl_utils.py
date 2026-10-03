"""Defines rl utilities."""
import abc
import collections
import numpy as np
import torch


def pad(episodes):
    """Pads episodes to all be the same length by repeating the last exp.

    Args:
        episodes (list[list[Experience]]): episodes to pad.

    Returns:
        padded_episodes (list[list[Experience]]): now of shape
            (batch_size, max_len)
        mask (torch.BoolTensor): of shape (batch_size, max_len) with value 0 for
            padded experiences.
    """
    max_len = max(len(episode) for episode in episodes)
    mask = torch.zeros((len(episodes), max_len), dtype=torch.bool)
    padded_episodes = []
    for i, episode in enumerate(episodes):
        padded = episode + [episode[-1]] * (max_len - len(episode))
        padded_episodes.append(padded)
        mask[i, :len(episode)] = True
    return padded_episodes, mask


class Experience(collections.namedtuple(
        "Experience", ("state", "action", "reward", "next_state", "done",
                       "info", "agent_state", "next_agent_state"))):
    """Defines a single (s, a, r, s')-tuple.

    Includes the agent state, as well for any agents with hidden state.
    """