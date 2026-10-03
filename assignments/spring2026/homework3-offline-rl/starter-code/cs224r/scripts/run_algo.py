# =============================================================================
# run_algo.py  —  Train an offline RL agent (AWAC, or IQL) or a BC agent on a 
# specified environment
# =============================================================================
#
# FILE STRUCTURE
# --------------
# Trainer (class)
#   __init__()        : selects the agent class (BCAgent / AWACAgent / IQLAgent)
#                       based on params['algo'], builds the agent_params dict, 
#                       then instantiates RL_Trainer.
#   run_training_loop(): delegates to RL_Trainer.run_training_loop().
#
# main() (function)
#   - Defines all command-line arguments (--algo, --env_name, etc.).
#   - Instantiates Trainer and calls run_training_loop().
# =============================================================================

import argparse
import os
import time

from cs224r.infrastructure.utils import get_env_kwargs
from cs224r.infrastructure.rl_trainer_awac import RL_Trainer


class Trainer(object):

    def __init__(self, params):
        self.params = params

        algo = params['algo']

        if algo == 'bc':
            from cs224r.agents.bc_agent import BCAgent
            agent_class = BCAgent
            train_args = {
                'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
                'num_critic_updates_per_agent_update': 1,
                'train_batch_size': params['batch_size'],
                'double_q': False,
            }
        elif algo == 'awac':
            from cs224r.agents.awac_agent import AWACAgent
            agent_class = AWACAgent
            train_args = {
                'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
                'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
                'train_batch_size': params['batch_size'],
                'double_q': params['double_q'],
            }
        elif algo == 'iql':
            from cs224r.agents.iql_agent import IQLAgent
            agent_class = IQLAgent
            train_args = {
                'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
                'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
                'train_batch_size': params['batch_size'],
                'double_q': params['double_q'],
            }
        else:
            raise ValueError(f"Unknown algo: {algo}")

        env_args = get_env_kwargs(params['env_name'])
        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = agent_class
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(self.agent_params['num_timesteps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=('bc', 'awac', 'iql'),
                        help='Which algorithm to run: bc, awac, or iql')
    parser.add_argument('--env_name', default='antmaze-umaze-v0',
                        help='Environment name')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=2)

    # --- dataset ---
    parser.add_argument('--offline_dataset', type=str, default=None,
                        help='Path to a .npz offline dataset to pre-load into the replay buffer')
    parser.add_argument('--filter_top_percent', type=float, default=None,
                        help='Only use the top X%% trajectories (by episode return) from the offline dataset.')
    parser.add_argument('--rew_shift', type=float, default=0.0)
    parser.add_argument('--rew_scale', type=float, default=1.0)

    # --- algo-specific hyperparameters ---
    parser.add_argument('--awac_lambda', type=float, default=0.1)
    parser.add_argument('--iql_expectile', type=float, default=0.9)

    # --- network architecture ---
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_actions', type=int, default=10)

    # --- training ---
    parser.add_argument('--num_timesteps', type=int, default=150000)
    parser.add_argument('--batch_size', type=int, default=256)

    # --- hardware ---
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # --- logging & checkpoint ---
    parser.add_argument('--scalar_log_freq', type=int, default=int(2e3))
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--checkpoint_interval', type=int, default=-1,
                        help='Save policy checkpoint every N steps (-1 to disable)')
    parser.add_argument('--buffer_interval', type=int, default=-1,
                        help='Save replay buffer as .npz every N steps (-1 to disable)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to a checkpoint_step{N}/ directory to restore policy and critics')

    args = parser.parse_args()
    params = vars(args)

    params['double_q'] = True
    params['num_agent_train_steps_per_iter'] = 1
    params['num_critic_updates_per_agent_update'] = 1

    _D4RL_EP_LENS = {
        'antmaze': 700,
        'halfcheetah': 1000,
        'hopper': 1000,
        'walker': 1000,
        'maze2d': 300,
    }
    _is_d4rl = any(k in params['env_name'] for k in _D4RL_EP_LENS)

    if params['env_name'] == 'PointmassEasy-v0':
        params['ep_len'] = 50
        params['num_timesteps'] = 50000
        params['scalar_log_freq'] = 2000
        params['n_eval_episodes'] = 20
    elif params['env_name'] == 'PointmassMedium-v0':
        params['ep_len'] = 150
        params['num_timesteps'] = 50000
        params['scalar_log_freq'] = 2000
        params['n_eval_episodes'] = 20
    elif params['env_name'] == 'PointmassHard-v0':
        params['ep_len'] = 200
        params['num_timesteps'] = 50000
        params['scalar_log_freq'] = 2000
        params['n_eval_episodes'] = 20
    elif params['env_name'] == 'PointmassVeryHard-v0':
        params['ep_len'] = 200
        params['num_timesteps'] = 50000
        params['scalar_log_freq'] = 2000
        params['n_eval_episodes'] = 20
    elif _is_d4rl:
        for k, v in _D4RL_EP_LENS.items():
            if k in params['env_name']:
                params['ep_len'] = v
                break
        if 'antmaze-umaze' in params['env_name']:
            params['num_timesteps'] = 300000
            params['scalar_log_freq'] = 50000
        if 'antmaze-medium' in params['env_name']:
            params['num_timesteps'] = 300000
            params['scalar_log_freq'] = 50000
        params['n_eval_episodes'] = 50

    # D4RL: load dataset from the env
    if _is_d4rl and not params.get('offline_dataset'):
        params['load_d4rl_dataset'] = True

    logdir_prefix = 'hw3_'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    os.makedirs(data_path, exist_ok=True)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    os.makedirs(logdir, exist_ok=True)
    params['logdir'] = logdir

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
