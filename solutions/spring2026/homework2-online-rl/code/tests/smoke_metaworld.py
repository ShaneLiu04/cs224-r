"""Minimal Meta-World runtime smoke test for AutoDL."""

import numpy as np

import mw


def main():
    environment = mw.make()
    time_step = environment.reset()
    assert time_step.observation.shape == environment.observation_spec().shape
    action = np.zeros(environment.action_spec().shape, dtype=np.float32)
    next_time_step = environment.step(action)
    assert np.isfinite(next_time_step.observation).all()
    print(
        "Meta-World smoke passed:",
        time_step.observation.shape,
        environment.action_spec().shape,
    )


if __name__ == "__main__":
    main()
