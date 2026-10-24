"""Main script to setup goal-conditioned RL experiments."""
import argparse
import random

import numpy as np
import torch
from torch.utils import tensorboard

import bit_flip_env

import trainer
from utils import HERType


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Flip some bits!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--env', type=str, default='bit_flip',
                        help='choose between bit_flip_env and sawyer_reach')
    parser.add_argument('--num_bits', type=int, default=7,
                        help='number of bits in the bit flipping environment')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs Q-learning is run for')
    parser.add_argument('--her_type', type=str, default='no_hindsight',
                        help='type of HER to use')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='manual seed for pytorch')
    parser.add_argument('--num_relabeled', type=int, default=4,
                        help='relabeled transitions per transition (future/random HER)')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.env == 'bit_flip':
        env = bit_flip_env.BitFlipEnv(args.num_bits)
        input_dim = 2 * args.num_bits
        action_dim = args.num_bits
        steps_per_episode = args.num_bits
        env_reward_function = lambda x, y: 0.0 if np.array_equal(x, y) else -1.0
        tensorboard_log_dir = (
            f'./logs/gcrl/bit_flip/num_bits:{args.num_bits}/HER_type:{args.her_type}/seed:{args.random_seed}' # pylint: disable=line-too-long
        ) if args.log_dir is None else args.log_dir

    elif args.env == 'sawyer_reach':
        # mujoco-py / multiworld are only needed for Sawyer, so bit-flip
        # runs work in environments without MuJoCo.
        import gym
        import multiworld
        import sawyer_action_discretize
        multiworld.register_all_envs()
        env = gym.make('SawyerReachXYEnv-v1')
        env = sawyer_action_discretize.SawyerActionDiscretize(
            env, render_every_step=False)
        input_dim = 4
        action_dim = 4
        steps_per_episode = 50
        env_reward_function = lambda x, y: -np.linalg.norm(x - y)
        tensorboard_log_dir = (
            f'./logs/gcrl/sawyer_reach/HER_type:{args.her_type}/seed:{args.random_seed}' # pylint: disable=line-too-long
        ) if args.log_dir is None else args.log_dir

    print(f'logging experiment at: {tensorboard_log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=tensorboard_log_dir)
    trainer.train(
        env=env,
        input_dim=input_dim,
        action_dim=action_dim,
        env_reward_function=env_reward_function,
        num_epochs=args.num_epochs,
        steps_per_episode=steps_per_episode,
        her_type=HERType[args.her_type.upper()],
        num_relabeled=args.num_relabeled,
        writer=writer,
    )
