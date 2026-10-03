"""Flappy Bird Gymnasium environment with easy and hard difficulty modes.

This file is fully provided -- do not modify.

Easy mode: each pipe has one gap opening.
Hard mode: each pipe has two gap openings (bimodal expert demonstrations).

Action space: target y position (normalised 0-1).
The environment uses a PD controller to convert the target position into
thrust, creating momentum-based physics. With moderate PD gains the bird
tracks slowly, so faster pipes require more anticipation.

Observation is always 4-D: [dist_to_pipe, gap1_y, gap2_y, bird_y] (normalised).
bird_vel is tracked internally but NOT included in the observation.
"""

import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W = 800
SCREEN_H = 448  # Divisible by 16 for video encoding

BIRD_X = 80
BIRD_RADIUS = 15

# Physics constants
GRAVITY = 0.5          # pixels/step^2 downward (positive = down)
THRUST_SCALE = 2.5     # pixels/step^2 per unit action (positive action = up)
MAX_VEL = 10.0         # velocity clamp

# PD controller gains (responsive — bird tracks target quickly)
PD_KP = 3.5            # proportional gain on position error
PD_KD = 1.2            # derivative gain on velocity
HOVER_THRUST = GRAVITY / THRUST_SCALE  # thrust to counteract gravity (~0.333)

PIPE_WIDTH = 60
PIPE_GAP_SIZE = 75   # vertical size of each opening
PIPE_SPEED = 3.0
PIPE_SPACING = 200   # horizontal distance between consecutive pipes

MAX_STEPS = 1000

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

HARD_GAP_SEPARATION = 150.0  # vertical distance between the two gap centers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_easy_pipe(rng: np.random.Generator):
    margin = PIPE_GAP_SIZE // 2 + 20
    gap_y = rng.integers(margin, SCREEN_H - margin)
    return float(gap_y), float(gap_y)


def _hard_pipe(rng: np.random.Generator):
    """Generate two gap positions separated by a fixed distance."""
    margin = PIPE_GAP_SIZE // 2 + 20
    max_upper = SCREEN_H - margin - HARD_GAP_SEPARATION
    gap1_y = float(rng.integers(margin, int(max_upper)))
    gap2_y = gap1_y + HARD_GAP_SEPARATION
    return gap1_y, gap2_y


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FlappyBirdEnv(gym.Env):
    """Flappy Bird with PD-controlled target-position action space.

    Action: value in [0, 1] = normalised target y position.
    The environment internally uses a PD controller to convert the target
    into thrust, creating momentum-based dynamics.  Moderate PD gains mean
    the bird tracks with some lag — faster pipes need more anticipation.

    Observation (4-D): [dist_to_pipe, gap1_y, gap2_y, bird_y] (normalised).
    bird_vel is tracked internally but NOT included in the observation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, difficulty: str = "easy", render_mode: Optional[str] = None,
                 pipe_speed: float = 3.0):
        super().__init__()
        assert difficulty in ("easy", "hard")
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.pipe_speed = pipe_speed

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -0.5], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 1.5], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(0.0), high=np.float32(1.0), shape=(1,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._screen = None
        self._clock = None
        self._sprites_loaded = False

    # ---- gym API ----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.bird_y = SCREEN_H / 2.0
        self.bird_vel = 0.0
        self.score = 0
        self.step_count = 0
        self._pipe_count = 0
        self._next_hard_gaps = None

        self.pipes = []
        self._spawn_pipe(SCREEN_W + 80)
        self._spawn_pipe(SCREEN_W + 80 + PIPE_SPACING)

        return self._get_obs(), {}

    def step(self, action):
        # Action = target y position (normalised 0-1)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        target_y_norm = float(np.clip(action_arr[0], 0.0, 1.0))

        # PD controller: convert target position → thrust
        # In screen coords: y=0 is top, y=SCREEN_H is bottom
        # Positive thrust → bird goes UP (decreases y)
        bird_y_norm = self.bird_y / SCREEN_H
        error = target_y_norm - bird_y_norm           # positive = target is below
        vel_norm = self.bird_vel / MAX_VEL            # positive = moving down
        # target below → less thrust (let gravity pull down) → subtract KP*error
        # moving down → more thrust (dampen) → add KD*vel_norm
        thrust = HOVER_THRUST - PD_KP * error + PD_KD * vel_norm
        thrust = float(np.clip(thrust, -1.0, 1.0))

        # Apply thrust to momentum physics
        self.bird_vel += GRAVITY - thrust * THRUST_SCALE
        self.bird_vel = float(np.clip(self.bird_vel, -MAX_VEL, MAX_VEL))
        self.bird_y += self.bird_vel

        for p in self.pipes:
            p["x"] -= self.pipe_speed

        if self.pipes and self.pipes[0]["x"] < -PIPE_WIDTH:
            self.pipes.pop(0)
        if len(self.pipes) < 3:
            last_x = self.pipes[-1]["x"] if self.pipes else SCREEN_W
            self._spawn_pipe(last_x + PIPE_SPACING)

        reward = 0.0
        for p in self.pipes:
            if not p["scored"] and p["x"] + PIPE_WIDTH / 2 < BIRD_X:
                p["scored"] = True
                self.score += 1
                reward = 1.0

        terminated = self._check_collision()
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        if terminated:
            reward = -1.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return None
        return self._render_pygame()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    # ---- internals --------------------------------------------------------

    def _spawn_pipe(self, x: float):
        if self.difficulty == "easy":
            g1, g2 = _random_easy_pipe(self._rng)
        else:
            if self._pipe_count % 2 == 0:
                if self._next_hard_gaps is not None:
                    g1, g2 = self._next_hard_gaps
                    self._next_hard_gaps = None
                else:
                    g1, g2 = _hard_pipe(self._rng)
            else:
                upcoming_g1, upcoming_g2 = _hard_pipe(self._rng)
                self._next_hard_gaps = (upcoming_g1, upcoming_g2)
                mid = (upcoming_g1 + upcoming_g2) / 2.0
                g1, g2 = mid, mid
            self._pipe_count += 1
        self.pipes.append({"x": x, "gap1": g1, "gap2": g2, "scored": False})

    def _get_obs(self):
        nxt = None
        for p in self.pipes:
            if p["x"] + PIPE_WIDTH / 2 > BIRD_X:
                nxt = p
                break
        if nxt is None:
            nxt = self.pipes[-1]

        return np.array([
            (nxt["x"] - BIRD_X) / SCREEN_W,
            nxt["gap1"] / SCREEN_H,
            nxt["gap2"] / SCREEN_H,
            self.bird_y / SCREEN_H,
        ], dtype=np.float32)

    def _check_collision(self):
        if self.bird_y - BIRD_RADIUS < 0 or self.bird_y + BIRD_RADIUS > SCREEN_H:
            return True
        for p in self.pipes:
            px = p["x"]
            if not (BIRD_X + BIRD_RADIUS > px - PIPE_WIDTH / 2 and
                    BIRD_X - BIRD_RADIUS < px + PIPE_WIDTH / 2):
                continue
            if self._bird_in_gap(p):
                continue
            return True
        return False

    def _bird_in_gap(self, pipe):
        half = PIPE_GAP_SIZE / 2
        by = self.bird_y
        if pipe["gap1"] - half <= by <= pipe["gap1"] + half:
            return True
        if self.difficulty == "hard":
            if pipe["gap2"] - half <= by <= pipe["gap2"] + half:
                return True
        return False

    # ---- rendering --------------------------------------------------------

    def _load_sprites(self):
        import pygame
        self._bg = pygame.image.load(os.path.join(ASSET_DIR, "background2.png"))
        self._bg = pygame.transform.scale(self._bg, (SCREEN_W, SCREEN_H))
        self._bird_up = pygame.image.load(os.path.join(ASSET_DIR, "robobird_up.png"))
        self._bird_up = pygame.transform.scale(
            self._bird_up, (BIRD_RADIUS * 3, BIRD_RADIUS * 3))
        self._bird_down = pygame.image.load(os.path.join(ASSET_DIR, "robobird_down.png"))
        self._bird_down = pygame.transform.scale(
            self._bird_down, (BIRD_RADIUS * 3, BIRD_RADIUS * 3))
        self._pipe_img = pygame.image.load(os.path.join(ASSET_DIR, "pipe.png"))
        self._sprites_loaded = True

    def _render_pygame(self):
        import pygame
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                pygame.display.set_caption("Flappy Bird")
            else:
                self._screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self._clock = pygame.time.Clock()
        if not self._sprites_loaded:
            self._load_sprites()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        self._screen.blit(self._bg, (0, 0))

        for p in self.pipes:
            self._draw_pipe(p)

        bird_sprite = self._bird_down if self.bird_vel > 0 else self._bird_up
        self._screen.blit(bird_sprite,
                          (BIRD_X - BIRD_RADIUS * 1.5, self.bird_y - BIRD_RADIUS * 1.5))

        font = pygame.font.SysFont(None, 36)
        score_surf = font.render(str(self.score), True, (255, 255, 255))
        self._screen.blit(score_surf, (SCREEN_W // 2 - 10, 20))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2))

    def _draw_pipe(self, pipe):
        import pygame
        px = int(pipe["x"] - PIPE_WIDTH / 2)
        half = PIPE_GAP_SIZE // 2

        if self.difficulty == "easy":
            gap_top = int(pipe["gap1"] - half)
            gap_bot = int(pipe["gap1"] + half)
            if gap_top > 0:
                s = pygame.transform.scale(self._pipe_img, (PIPE_WIDTH, gap_top))
                s = pygame.transform.flip(s, False, True)
                self._screen.blit(s, (px, 0))
            bot_h = SCREEN_H - gap_bot
            if bot_h > 0:
                s = pygame.transform.scale(self._pipe_img, (PIPE_WIDTH, bot_h))
                self._screen.blit(s, (px, gap_bot))
        else:
            g1_top = int(pipe["gap1"] - half)
            g1_bot = int(pipe["gap1"] + half)
            g2_top = int(pipe["gap2"] - half)
            g2_bot = int(pipe["gap2"] + half)
            if g1_top > 0:
                s = pygame.transform.scale(self._pipe_img, (PIPE_WIDTH, g1_top))
                s = pygame.transform.flip(s, False, True)
                self._screen.blit(s, (px, 0))
            mid_h = g2_top - g1_bot
            if mid_h > 0:
                s = pygame.transform.scale(self._pipe_img, (PIPE_WIDTH, mid_h))
                self._screen.blit(s, (px, g1_bot))
            bot_h = SCREEN_H - g2_bot
            if bot_h > 0:
                s = pygame.transform.scale(self._pipe_img, (PIPE_WIDTH, bot_h))
                self._screen.blit(s, (px, g2_bot))
