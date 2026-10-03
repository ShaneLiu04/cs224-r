# =============================================================================
# rl_trainer_awac.py — Training loop for offline RL agents (AWAC, IQL, BC)
# =============================================================================
# FILE STRUCTURE
# --------------
#   class RL_Trainer
#     __init__                   — env creation, agent instantiation, offline dataset loading
#
#     Dataset loading
#       load_offline_dataset     — load transitions from a .npz file into the replay buffer;
#                                  supports optional top-% trajectory filtering
#       load_d4rl_dataset        — load D4RL benchmark dataset with reward normalization
#
#     Dataset utilities
#       _filter_top_trajectories — keep only top-X% episodes by return
#       _plot_dataset_ep_lengths — save a histogram of episode lengths to disk
#
#     Training loop
#       run_training_loop        — main loop over n_iter: train, log, checkpoint
#       train_agent              — inner loop: sample batch, call agent.train()
#
#     Evaluation & logging
#       perform_logging          — collect eval rollouts, compute returns/normalized scores,
#                                  log to stdout and optionally W&B
#       _save_eval_video         — render one eval rollout and save as .mp4
#
#     Checkpointing
#       save_policy_checkpoint   — save policy + critic weights to disk
#       load_checkpoint          — restore policy + critic from a checkpoint directory
#       save_replay_buffer       — serialize replay buffer to .npz; plot stats/trajectories
#       _critic_state_dict       — build a serializable dict from a critic (handles twins/V-net)
#       _load_critic_state_dict  — restore critic weights from that dict
#
#     Visualization
#       _visualize_trajectories  — 2-D pointmass only: color-coded trajectory plot

from collections import OrderedDict
import os
import sys
import time

import gym
from gym import wrappers
import matplotlib
import numpy as np
import torch
import wandb

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure import utils
from cs224r.infrastructure.wrappers import ReturnWrapper

from cs224r.agents.iql_agent import IQLAgent
from cs224r.agents.awac_agent import AWACAgent
from cs224r.agents.bc_agent import BCAgent
from cs224r.infrastructure.utils import register_custom_envs

import d4rl

class RL_Trainer(object):
    """Training loop for offline and online RL agents (AWAC, IQL, BC)."""

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.use_wandb = self.params.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project='HW3_OfflineRL_' + self.params.get('env_name', 'env') + '-modal-v1',
                name=self.params.get('exp_name', 'run'),
                config=self.params,
            )

        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        register_custom_envs()
        env_name = self.params['env_name']
        self.eval_env = gym.make(env_name)
        self.env = gym.make(env_name)

        if 'pointmass' in env_name.lower() and hasattr(self.env, 'set_logdir'):
            matplotlib.use('Agg')
            self.env.set_logdir(self.params['logdir'] + '/expl_')
            self.eval_env.set_logdir(self.params['logdir'] + '/eval_')

        video_log_freq = self.params.get('video_log_freq', -1)
        self.episode_trigger = (
            (lambda episode: episode % video_log_freq == 0)
            if video_log_freq > 0 else
            (lambda episode: False)
        )

        if 'env_wrappers' in self.params:
            _is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
            if _is_discrete:
                self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
                self.env = ReturnWrapper(self.env)
                self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.env = params['env_wrappers'](self.env)

            if _is_discrete:
                self.eval_env = wrappers.RecordEpisodeStatistics(self.eval_env, deque_size=1000)
                self.eval_env = ReturnWrapper(self.eval_env)
                self.eval_env = wrappers.RecordVideo(self.eval_env, os.path.join(self.params['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.eval_env = params['env_wrappers'](self.eval_env)

            self.mean_episode_reward = -float('nan')
        self.env.seed(seed)
        self.eval_env.seed(seed)

        if not self.params.get('ep_len'):
            spec_max = getattr(self.env.spec, 'max_episode_steps', None)
            self.params['ep_len'] = spec_max or 1000
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        if 'model' in dir(self.env):
            self.fps = min(30, int(1/self.env.model.opt.timestep))
        elif 'env_wrappers' in self.params:
            self.fps = 30  # Not actually used when using the Monitor wrapper
        elif hasattr(self.env, 'env') and hasattr(self.env.env, 'metadata') and \
                'video.frames_per_second' in self.env.env.metadata:
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

        if self.params.get('offline_dataset'):
            self.load_offline_dataset(self.params['offline_dataset'])

        if self.params.get('load_d4rl_dataset', False):
            self.load_d4rl_dataset()

        if self.params.get('load_checkpoint'):
            self.load_checkpoint(self.params['load_checkpoint'])

    def load_offline_dataset(self, dataset_path):
        """Pre-populate the replay buffer from a .npz file (obs/action/reward/done)."""
        print(f"\nLoading offline dataset from: {dataset_path}")
        data = np.load(dataset_path)
        obs     = data['obs']
        rewards = data['reward'].astype(np.float32)
        dones   = data['done']

        buf = self.agent.replay_buffer
        continuous = getattr(buf, 'continuous', False)

        if continuous:
            actions = data['action'].astype(np.float32)
        else:
            actions = data['action'].astype(np.int32)

        # --- Filter by top-X% trajectories (by episode return) ---
        top_pct = self.params.get('filter_top_percent', None)
        if top_pct is not None:
            self._log_dataset_stats(rewards, dones, label="Before filtering")
            obs, actions, rewards, dones = self._filter_top_trajectories(
                obs, actions, rewards, dones, top_pct)
            self._log_dataset_stats(rewards, dones,
                                    label=f"After filtering (top {top_pct}%)")

        n = min(len(obs), buf.size)

        if buf.obs is None:
            buf.obs    = np.empty([buf.size] + list(obs.shape[1:]), dtype=np.float32)
            if continuous:
                ac_shape = [buf.size] + list(actions.shape[1:]) if actions.ndim > 1 else [buf.size]
                buf.action = np.empty(ac_shape, dtype=np.float32)
            else:
                buf.action = np.empty([buf.size], dtype=np.int32)
            buf.reward = np.empty([buf.size], dtype=np.float32)
            buf.done   = np.empty([buf.size], dtype=bool)

        buf.obs[:n]       = obs[:n]
        buf.action[:n]    = actions[:n]
        buf.reward[:n]    = rewards[:n]
        buf.done[:n]      = dones[:n]
        buf.num_in_buffer = n
        buf.next_idx      = n % buf.size

        print(f"Loaded {n} transitions into replay buffer.")

        algo_name = type(self.agent).__name__.replace('Agent', '').lower()
        self._plot_dataset_ep_lengths(
            rewards=rewards[:n],
            dones=dones[:n],
            algo_name=algo_name,
            low_pct=-1,
            top_pct=top_pct    if top_pct    is not None else -1,
        )

    @staticmethod
    def _plot_dataset_ep_lengths(rewards, dones, algo_name, low_pct, top_pct):
        """Save a histogram of per-episode lengths for the loaded dataset.

        Saved to: vis/dataset/{algo_name}/lowpct{low_pct}_toppct{top_pct}.png
        relative to the current working directory.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Compute episode lengths from done flags
        traj_slices, traj_returns = RL_Trainer._compute_traj_returns(rewards, dones)
        ep_lengths = np.array([end - start for start, end in traj_slices])

        if ep_lengths.size == 0:
            print("[Warning] No complete episodes found in dataset; skipping ep-length plot.")
            return

        # Build save path
        save_dir = os.path.join('vis', 'dataset', algo_name)
        os.makedirs(save_dir, exist_ok=True)
        fname = f'lowpct{low_pct}_toppct{top_pct}.png'
        save_path = os.path.join(save_dir, fname)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ep_lengths, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.axvline(ep_lengths.mean(),    color='tomato',     linewidth=1.5,
                   linestyle='--', label=f'mean={ep_lengths.mean():.1f}')
        ax.axvline(np.median(ep_lengths), color='gold',      linewidth=1.5,
                   linestyle='--', label=f'median={np.median(ep_lengths):.1f}')
        ax.set_xlabel('Episode length (steps)')
        ax.set_ylabel('Count')
        ax.set_title(
            f'Dataset episode lengths  |  {algo_name.upper()}  |  '
            f'{len(ep_lengths)} episodes  |  '
            f'low_pct={low_pct}  top_pct={top_pct}'
        )
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [Dataset vis] Episode-length histogram saved to: {save_path}")

    @staticmethod
    def _compute_traj_returns(rewards, dones):
        """Return (traj_slices, traj_returns) for a flat transition array."""
        traj_slices = []
        traj_returns = []
        ep_start = 0
        ep_return = 0.0
        for i in range(len(rewards)):
            ep_return += rewards[i]
            if dones[i]:
                traj_slices.append((ep_start, i + 1))
                traj_returns.append(ep_return)
                ep_start = i + 1
                ep_return = 0.0
        if ep_start < len(rewards):   # trailing incomplete episode
            traj_slices.append((ep_start, len(rewards)))
            traj_returns.append(ep_return)
        return traj_slices, np.array(traj_returns, dtype=np.float32)

    @staticmethod
    def _log_dataset_stats(rewards, dones, label="Dataset"):
        _, traj_returns = RL_Trainer._compute_traj_returns(rewards, dones)
        n_trajs = len(traj_returns)
        n_transitions = len(rewards)
        print(f"\n  [{label}]")
        print(f"    Trajectories : {n_trajs}")
        print(f"    Transitions  : {n_transitions}")
        print(f"    Traj return  : mean={traj_returns.mean():.3f}  "
              f"max={traj_returns.max():.3f}  min={traj_returns.min():.3f}")

    @staticmethod
    def _filter_top_trajectories(obs, actions, rewards, dones, top_percent):
        """Return obs/actions/rewards/dones containing only the top-X% trajectories
        (ranked by episode return, highest first).  Trajectory boundaries are
        determined by the done flags in the dataset."""
        traj_slices, traj_returns = RL_Trainer._compute_traj_returns(rewards, dones)

        n_trajs = len(traj_slices)
        n_keep = max(1, int(np.ceil(n_trajs * top_percent / 100.0)))

        # Rank by return (descending); keep top-n_keep in temporal order.
        sorted_idxs = np.argsort(traj_returns)[::-1]
        keep_idxs = sorted(sorted_idxs[:n_keep])

        idx_lists = [np.arange(traj_slices[i][0], traj_slices[i][1])
                     for i in keep_idxs]
        sel = np.concatenate(idx_lists)
        return obs[sel], actions[sel], rewards[sel], dones[sel]

    @staticmethod
    def _split_into_trajectories(observations, actions, rewards, masks,
                                 dones_float, next_observations):
        trajs = [[]]
        for i in range(len(observations)):
            trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                              dones_float[i], next_observations[i]))
            if dones_float[i] == 1.0 and i + 1 < len(observations):
                trajs.append([])
        return trajs

    def load_d4rl_dataset(self):
        """Load D4RL offline dataset into the replay buffer."""
        print("\nLoading D4RL offline dataset from environment...")
        dataset = d4rl.qlearning_dataset(self.eval_env)

        lim = 1 - 1e-5
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        # Compute dones_float: 1 when trajectory boundary detected or terminal flag set
        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if (np.linalg.norm(dataset['observations'][i + 1] -
                               dataset['next_observations'][i]) > 1e-6
                    or dataset['terminals'][i] == 1.0):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        observations = dataset['observations'].astype(np.float32)
        actions = dataset['actions'].astype(np.float32)
        rewards = dataset['rewards'].astype(np.float32)
        masks = 1.0 - dataset['terminals'].astype(np.float32)
        dones_float = dones_float.astype(np.float32)
        next_observations = dataset['next_observations'].astype(np.float32)

        # Reward normalization
        env_name = self.params['env_name']
        if 'antmaze' in env_name:
            rewards -= 1.0
        elif ('halfcheetah' in env_name or 'walker2d' in env_name
              or 'hopper' in env_name):
            trajs = self._split_into_trajectories(
                observations, actions, rewards, masks, dones_float, next_observations)

            def compute_returns(traj):
                episode_return = 0
                for _, _, rew, _, _, _ in traj:
                    episode_return += rew
                return episode_return

            trajs.sort(key=compute_returns)
            rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
            rewards *= 1000.0

        buf = self.agent.replay_buffer
        n = len(observations)
        assert buf.size >= n, (
            "Dataset cannot be larger than replay buffer capacity "
            f"({n} > {buf.size})."
        )

        if buf.obs is None:
            buf.obs    = np.empty([buf.size] + list(observations.shape[1:]), dtype=np.float32)
            ac_shape   = [buf.size, actions.shape[1]] if actions.ndim > 1 else [buf.size]
            buf.action = np.empty(ac_shape, dtype=np.float32)
            buf.reward = np.empty([buf.size], dtype=np.float32)
            buf.done   = np.empty([buf.size], dtype=bool)

        buf.obs[:n]    = observations[:n]
        buf.action[:n] = actions[:n]
        buf.reward[:n] = rewards[:n]
        buf.done[:n]   = (masks[:n] == 0.0)  # True at terminal (mask=0)

        if buf.next_obs_explicit is None:
            buf.next_obs_explicit = np.empty(
                [buf.size] + list(next_observations.shape[1:]), dtype=np.float32)
        buf.next_obs_explicit[:n] = next_observations[:n]

        buf.num_in_buffer = n
        buf.next_idx      = n % buf.size

        n_eps = int(dones_float[:n].sum())
        print(f"Loaded {n} transitions ({n_eps} episodes) from D4RL dataset.")

    def run_training_loop(self, n_iter):
        """
        :param n_iter:  number of training iterations
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif (itr + 1) % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            envsteps_this_batch = 0

            self.total_envsteps += envsteps_this_batch

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            if isinstance(self.agent, (IQLAgent, AWACAgent)) and (itr % print_period == 0):
                self.dump_density_graphs(itr)

            # log/save
            if self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

            # save checkpoints/replay buffer
            t = self.agent.t
            ckpt_interval = self.params.get('checkpoint_interval', -1)
            buf_interval = self.params.get('buffer_interval', -1)
            if ckpt_interval > 0 and t > 0 and t % ckpt_interval == 0:
                self.save_policy_checkpoint(t)
            if buf_interval > 0 and t > 0 and t % buf_interval == 0:
                self.save_replay_buffer(t)

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def _save_eval_video(self):
        """Render one eval rollout and overwrite eval_traj_last.mp4 in the logdir."""
        try:
            import imageio
        except ImportError:
            print("[Warning] imageio not installed; skipping eval video. "
                  "Install with: pip install imageio[ffmpeg]")
            return

        paths = utils.sample_n_trajectories(
            self.eval_env, self.agent.eval_policy,
            ntraj=1, max_path_length=self.params['ep_len'],
            render=True, deterministic=True,
        )
        frames = paths[0].get('image_obs')
        if frames is None or frames.ndim < 4 or frames.shape[0] == 0:
            return  # render not available for this env

        video_path = os.path.join(self.params['logdir'], 'eval_traj_last.mp4')
        imageio.mimsave(video_path, frames, fps=self.fps)
        print(f"  [Eval video] saved to: {video_path}")

    def perform_logging(self, all_logs):
        last_log = all_logs[-1]

        logs = OrderedDict()
        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))

        # --- Discrete envs: training episode rewards from RecordEpisodeStatistics ---
        if hasattr(self, 'env') and hasattr(self.env, 'get_episode_rewards'):
            episode_rewards = self.env.get_episode_rewards()
            if len(episode_rewards) > 0:
                self.mean_episode_reward = np.mean(episode_rewards[-100:])
            if self.mean_episode_reward > -5000:
                logs["Train_AverageReturn"] = self.mean_episode_reward
                print("mean reward (100 episodes) %f" % self.mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        # --- Evaluation rollouts ---
        n_eval_episodes = self.params.get('n_eval_episodes', 50)
        eval_paths = utils.sample_n_trajectories(
            self.eval_env, self.agent.eval_policy,
            n_eval_episodes, self.params['ep_len'],
            deterministic=True)


        if 'antmaze' in self.params['env_name']:
            eval_returns = [(p["reward"] - 1.0).sum() for p in eval_paths]
        else:
            eval_returns = [p["reward"].sum() for p in eval_paths]
        eval_ep_lens = [len(p["reward"]) for p in eval_paths]

        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"]     = np.std(eval_returns)
        logs["Eval_MaxReturn"]     = np.max(eval_returns)
        logs["Eval_MinReturn"]     = np.min(eval_returns)
        logs["Eval_AverageEpLen"]  = np.mean(eval_ep_lens)

        logs['Buffer_Size'] = self.agent.replay_buffer.num_in_buffer

        # Overwrite eval_traj_last.mp4 with the latest rollout (continuous envs only).
        if not self.params['agent_params'].get('discrete', True):
            self._save_eval_video()

        sys.stdout.flush()
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
        print('Done logging...\n\n')

        if self.use_wandb:
            wandb.log(logs, step=self.agent.t)
            
    @staticmethod
    def _critic_state_dict(critic):
        ckpt = {
            'q_net':        critic.q_net.state_dict(),
            'q_net_target': critic.q_net_target.state_dict(),
        }
        # Twin Q (continuous AWAC)
        if hasattr(critic, 'q_net2'):
            ckpt['q_net2']        = critic.q_net2.state_dict()
            ckpt['q_net2_target'] = critic.q_net2_target.state_dict()
        # V-net (IQL)
        if hasattr(critic, 'v_net'):
            ckpt['v_net'] = critic.v_net.state_dict()
        return ckpt

    @staticmethod
    def _load_critic_state_dict(critic, ckpt):
        critic.q_net.load_state_dict(ckpt['q_net'])
        critic.q_net_target.load_state_dict(ckpt['q_net_target'])
        if 'q_net2' in ckpt and hasattr(critic, 'q_net2'):
            critic.q_net2.load_state_dict(ckpt['q_net2'])
            critic.q_net2_target.load_state_dict(ckpt['q_net2_target'])
        if 'v_net' in ckpt and hasattr(critic, 'v_net'):
            critic.v_net.load_state_dict(ckpt['v_net'])

    def save_policy_checkpoint(self, step):
        ckpt_dir = os.path.join(self.params['logdir'], f'checkpoint_step{step}')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.agent.eval_policy.save(os.path.join(ckpt_dir, 'policy.pt'))
        if hasattr(self.agent, 'critic'):
            torch.save(self._critic_state_dict(self.agent.critic),
                       os.path.join(ckpt_dir, 'critic.pt'))
        print(f"[Checkpoint] Saved to: {ckpt_dir}")

    def load_checkpoint(self, checkpoint_dir):
        """Restore policy and critics from a checkpoint directory."""
        print(f"\nLoading checkpoint from: {checkpoint_dir}")

        policy_path = os.path.join(checkpoint_dir, 'policy.pt')
        if os.path.exists(policy_path):
            self.agent.eval_policy.load_state_dict(
                torch.load(policy_path, map_location=ptu.device))
            print(f"  Policy loaded from: {policy_path}")

        expl_path = os.path.join(checkpoint_dir, 'critic.pt')
        if os.path.exists(expl_path):
            self._load_critic_state_dict(
                self.agent.critic,
                torch.load(expl_path, map_location=ptu.device))
            print(f"  Critic loaded from: {expl_path}")


    def save_replay_buffer(self, step):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        buf = self.agent.replay_buffer
        n = buf.num_in_buffer - 1  # last entry has no valid next_obs
        if n <= 0:
            return

        idxes = np.arange(n)
        next_idxes = (idxes + 1) % buf.size
        obs      = buf.obs[idxes]
        next_obs = buf.obs[next_idxes]
        actions  = buf.action[idxes]
        rewards  = buf.reward[idxes]
        dones    = buf.done[idxes]

        # Save .npz
        save_path = os.path.join(self.params['logdir'], f'replay_buffer_step{step}.npz')
        np.savez(save_path, obs=obs, action=actions, reward=rewards,
                 next_obs=next_obs, done=dones)

        # Compute per-episode returns and goal-reaching rate
        # Pointmass reward: 0 = goal reached, -1 = otherwise
        episode_returns = []
        episode_reached_goal = []
        cur = 0.0
        ep_reached = False
        for r, d in zip(rewards, dones):
            cur += r
            if r >= 0:  # reward == 0 means goal reached
                ep_reached = True
            if d:
                episode_returns.append(cur)
                episode_reached_goal.append(ep_reached)
                cur = 0.0
                ep_reached = False
        if cur != 0.0:
            episode_returns.append(cur)
            episode_reached_goal.append(ep_reached)

        print(f"\n[Buffer saved at step {step}] {n} transitions, "
              f"{len(episode_returns)} episodes")
        if episode_returns:
            goal_rate = np.mean(episode_reached_goal)
            print(f"  Episode return: mean={np.mean(episode_returns):.3f}  "
                  f"std={np.std(episode_returns):.3f}  "
                  f"min={np.min(episode_returns):.3f}  "
                  f"max={np.max(episode_returns):.3f}")
            print(f"  Goal reached: {goal_rate*100:.1f}%  "
                  f"({int(sum(episode_reached_goal))}/{len(episode_reached_goal)} episodes)")
        print(f"  Dataset saved to: {save_path}")

        # Visualize trajectories (2D Pointmass only)
        if obs.ndim == 2 and obs.shape[1] == 2:
            self._visualize_trajectories(obs, dones, episode_returns, step)

    def _visualize_trajectories(self, obs, dones, episode_returns, step):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Split obs into episodes at done boundaries
        episodes = []
        ep_start = 0
        for i in range(len(obs)):
            if dones[i] or i == len(obs) - 1:
                episodes.append(obs[ep_start:i + 1])
                ep_start = i + 1

        fig, ax = plt.subplots(figsize=(7, 7))
        ret_arr = np.array(episode_returns) if episode_returns else np.zeros(len(episodes))
        vmin, vmax = ret_arr.min(), ret_arr.max()
        norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-8))
        cmap = cm.viridis

        for i, ep in enumerate(episodes):
            color = cmap(norm(ret_arr[i]) if i < len(ret_arr) else 0.5)
            if len(ep) > 1:
                ax.plot(ep[:, 0], ep[:, 1], color=color, alpha=0.4, linewidth=0.6)
            ax.scatter(ep[0, 0], ep[0, 1], color='lime', s=8, zorder=5)
            ax.scatter(ep[-1, 0], ep[-1, 1], color='red', s=8, zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Buffer Trajectories  step={step}  n={len(episodes)} episodes')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Episode Return')

        viz_path = os.path.join(self.params['logdir'], f'trajectories_step{step}.jpg')
        fig.savefig(viz_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        print(f"  Trajectory visualization saved to: {viz_path}")

    def dump_density_graphs(self, itr):
        # Density graphs are only meaningful for 2-D pointmass envs.
        if 'pointmass' not in self.params['env_name'].lower():
            return

        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        filepath = lambda name: self.params['logdir']+'/curr_{}.png'.format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0: return

        H, xedges, yedges = np.histogram2d(states[:,0], states[:,1], range=[[0., 1.], [0., 1.]], density=True)
        plt.imshow(np.rot90(H), interpolation='bicubic')
        plt.colorbar()
        plt.title('State Density')
        self.fig.savefig(filepath('state_density'), bbox_inches='tight')

        plt.clf()
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)

        critic_values = self.agent.critic.qa_values(obs).mean(-1)
        critic_values = critic_values.reshape(ii.shape)
        plt.imshow(critic_values[::-1])
        plt.colorbar()
        plt.title('Predicted Critic Value')
        self.fig.savefig(filepath('critic_value'), bbox_inches='tight')

        plt.close('all')
