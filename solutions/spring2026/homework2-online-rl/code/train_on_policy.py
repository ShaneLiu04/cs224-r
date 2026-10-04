import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ["WANDB_DISABLE_CODE"] = "1"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import mw
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


# On policy rollout buffer
class RolloutBuffer:
    """Collects a fixed length on policy rollout, then yields it as one batch. """

    def __init__(self, rollout_length, obs_shape, action_shape, device):
        self.rollout_length = rollout_length
        self.device = device

        self.obs = np.zeros((rollout_length, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((rollout_length, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((rollout_length,), dtype=np.float32)
        self.discounts = np.zeros((rollout_length,), dtype=np.float32)
        self.next_obs = np.zeros((rollout_length, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((rollout_length,), dtype=np.float32)
        self.old_log_probs = np.zeros((rollout_length,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, discount, next_obs, done, old_log_prob):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = discount
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.old_log_probs[self.ptr] = old_log_prob
        self.ptr += 1
        if self.ptr >= self.rollout_length:
            self.full = True

    def ready(self):
        return self.full

    def get(self):
        """Return the entire rollout as a single-element iterable"""
        assert self.full, "Rollout buffer not yet full."
        data = (
            self.obs,
            self.actions,
            self.rewards,
            self.discounts,
            self.next_obs,
            self.dones,
            self.old_log_probs,
        )
        # Wrap in a list so agent.update() can iterate over it with a for loop
        return [tuple(torch.as_tensor(x, device=self.device) for x in data)]

    def reset(self):
        self.ptr = 0
        self.full = False


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_wandb=self.cfg.use_wandb,
                             wandb_project=self.cfg.wandb_project,
                             wandb_entity=self.cfg.wandb_entity,
                             wandb_group=self.cfg.wandb_group,
                             cfg=self.cfg)
        # create envs
        self.train_env = mw.make()
        self.eval_env = mw.make()

        # Demo buffer for BC pretraining
        obs_spec = self.train_env.observation_spec()
        action_spec = self.train_env.action_spec()
        data_specs = (obs_spec,
                      action_spec,
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.demo_storage = ReplayBufferStorage(data_specs,
                                                self.work_dir / 'demos')
        self.demo_loader = make_replay_loader(
            self.work_dir / 'demos', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._demo_iter = None

        from distutils.dir_util import copy_tree
        copy_tree("/root/demos/",
                  str(self.work_dir / 'demos'))

        # On policy rollout buffer
        self.rollout_buffer = RolloutBuffer(
            rollout_length=self.cfg.rollout_length, 
            obs_shape=obs_spec.shape,
            action_shape=action_spec.shape,
            device=self.device,
        )

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def demo_iter(self):
        if self._demo_iter is None:
            self._demo_iter = iter(self.demo_loader)
        return self._demo_iter

    def eval(self, num_eval_episodes=None):
        step, episode, total_reward, total_success = 0, 0, 0, 0
        if num_eval_episodes is None:
            num_eval_episodes = self.cfg.num_eval_episodes
        eval_until_episode = utils.Until(num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            total_success += time_step.reward > 0.0
            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_success', total_success / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('eval_total_time', self.timer.total_time())

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                       self.cfg.action_repeat)

        # BC pretraining
        for pretrain_step in range(self.cfg.pretrain_steps):
            metrics = self.agent.bc(self.demo_iter)
            self.logger.log_metrics(metrics, pretrain_step, ty='pretrain')
        self.agent.set_reference_policy()

        # On-policy training loop
        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        metrics = None

        while train_until_step(self.global_step):

            # Episode boundary
            if time_step.last():
                self._global_episode += 1
                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(self.global_frame,
                                                  ty='train') as log:
                    log('fps', episode_frame / elapsed_time)
                    log('total_time', total_time)
                    log('episode_reward', episode_reward)
                    log('episode_length', episode_frame)
                    log('episode', self.global_episode)
                    log('step', self.global_step)

                time_step = self.train_env.reset()
                episode_step = 0
                episode_reward = 0

                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # Eval
            if eval_every_step(self.global_step):
                self.eval()

            # Collect one transition and store log prob
            obs = time_step.observation
            with torch.no_grad(), utils.eval_mode(self.agent):
                obs_t  = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                dist = self.agent.actor(obs_t)
                action_t = dist.sample(clip=None)
                # Sum over action dims to get a scalar log prob per transition
                old_log_prob = dist.log_prob(action_t).sum(-1).item()
                action = action_t.cpu().numpy()[0]

            next_time_step  = self.train_env.step(action)
            done = float(next_time_step.last())
            reward   = next_time_step.reward
            discount = next_time_step.discount * self.cfg.discount

            self.rollout_buffer.add(
                obs, action, reward, discount,
                next_time_step.observation, done, old_log_prob)

            episode_reward += reward
            episode_step += 1
            self._global_step += 1
            time_step = next_time_step

            # PPO update once rollout is full
            if self.rollout_buffer.ready():
                ppo_metrics = self.agent.update(self.rollout_buffer.get())
                self.logger.log_metrics(ppo_metrics, self.global_frame, ty='actor')

                # Optional: interleave BC to stay close to demo distribution
                if self.global_frame > 0 and self.cfg.bc_freq > 0 \
                        and self.global_step % self.cfg.bc_freq == 0:
                    bc_metrics = self.agent.bc(self.demo_iter)
                    self.logger.log_metrics(bc_metrics, self.global_frame, ty='actor')

                self.rollout_buffer.reset()

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='on_policy_config')
def main(cfg):
    from train_on_policy import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
