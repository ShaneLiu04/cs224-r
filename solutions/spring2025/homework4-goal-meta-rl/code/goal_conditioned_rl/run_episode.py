"""Run the Q-network on the environment for fixed steps.

Complete the code marked TODO."""
import numpy as np # pylint: disable=unused-import
import torch # pylint: disable=unused-import


def run_episode(
    env,
    q_net, # pylint: disable=unused-argument
    steps_per_episode,
):
    """Runs the current policy on the given environment.

    Args:
        env (gym): environment to generate the state transition
        q_net (QNetwork): Q-Network used for computing the next action
        steps_per_episode (int): number of steps to run the policy for

    Returns:
        episode_experience (list): list containing the transitions
                        (state, action, reward, next_state, goal_state)
        episodic_return (float): reward collected during the episode
        succeeded (bool): DQN succeeded to reach the goal state or not
    """

    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False
    episodic_return = 0.0

    # reset the environment to get the initial state
    state, goal_state = env.reset() # pylint: disable=unused-variable

    for _ in range(steps_per_episode):

        # ======================== TODO modify code ========================
        inputs = np.concatenate([state, goal_state]).astype(np.float32)
        with torch.no_grad():
            q_values = q_net(torch.from_numpy(inputs).unsqueeze(0))
        action = int(torch.argmax(q_values, dim=-1).item())

        next_state, reward, done, info = env.step(action)

        episode_experience.append(
            (state.copy(), action, reward, next_state.copy(), goal_state.copy()))
        episodic_return += float(reward)
        state = next_state

        succeeded = bool(info['successful_this_state'])
        if done:
            break
        # ========================      END TODO       ========================

    return episode_experience, episodic_return, succeeded
