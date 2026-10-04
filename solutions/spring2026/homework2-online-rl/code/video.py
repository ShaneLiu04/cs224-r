import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        """Record evaluation rollouts as image sequences that can be written to disk."""
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        """Start a new recording and optionally capture the initial frame."""
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        """Append one rendered environment frame if recording is enabled."""
        if self.enabled:
            frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        """Write the recorded evaluation frames to a video file."""
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        """Record training observations by converting them into displayable frames."""
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        """Start a new training recording and optionally add the first frame."""
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        """Resize and append one observation frame if recording is enabled."""
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        """Write the recorded training frames to a video file."""
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
