"""Tensorboard and padding utilities."""
import os

import torch
from torch.utils import tensorboard


class EpisodeAndStepWriter(object):
    """Logs to tensorboard against both episode and number of steps."""

    def __init__(self, log_dir):
        self._episode_writer = tensorboard.SummaryWriter(
                os.path.join(log_dir, "episode"))
        self._step_writer = tensorboard.SummaryWriter(
                os.path.join(log_dir, "step"))

    def add_scalar(self, key, value, episode, step):
        self._episode_writer.add_scalar(key, value, episode)
        self._step_writer.add_scalar(key, value, step)
