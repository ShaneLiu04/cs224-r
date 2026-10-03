# =============================================================================
# iql_agent.py  —  IQL agent (offline RL via implicit Q-learning)
# =============================================================================
# FILE STRUCTURE
# --------------
# IQLAgent
#   __init__          : sets up the following components:
#                         - self.critic  : IQLCritic (Q + V networks)
#                         - self.actor   : MLPPolicyAWAC
#
#   estimate_advantage() : computes A(s,a) = Q(s,a) - V(s).
#
#   train()           : called every step by RL_Trainer.  When enough data is
#                       available it performs:
#                         1. Update V-network  (IQLCritic.update_v).
#                         2. Estimate advantages and update actor.
#                         3. Update Q-network  (IQLCritic.update_q).
#                         4. Soft-update target network.
#
# WHERE TO EDIT
# -------------
# * estimate_advantage()  —  Fill in ### YOUR CODE START/END ###.
#
# * train()               —  Fill in ### YOUR CODE START/END ###.
# =============================================================================

import torch

from cs224r.critics.iql_critic import IQLCritic
from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure.utils import MemoryOptimizedReplayBuffer
from cs224r.policies.MLP_policy import MLPPolicyAWAC
from .base_agent import BaseAgent


class IQLAgent(BaseAgent):
    """Offline RL agent using Implicit Q-Learning (IQL)."""

    def __init__(self, env, agent_params):
        super(IQLAgent, self).__init__(env, agent_params)

        self.discrete_action = agent_params.get('discrete', True)
        ac_dim = agent_params['ac_dim']

        buf_size = agent_params.get('replay_buffer_size', 1000000)
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            buf_size, 1, float_obs=True,
            continuous_action=not self.discrete_action, ac_dim=ac_dim
        )

        self.critic = IQLCritic(agent_params, self.optimizer_spec)

        self.actor = MLPPolicyAWAC(
            agent_params['ac_dim'],
            agent_params['ob_dim'],
            agent_params['n_layers'],
            agent_params['size'],
            agent_params['discrete'],
            agent_params['learning_rate'],
            lambda_awac=agent_params['awac_lambda'],
            max_grad_norm=agent_params.get('grad_norm_clipping', 1.0),
            max_steps=agent_params.get('num_timesteps', None),
        )

        self.eval_policy = self.actor
        self.rew_shift = agent_params['rew_shift']
        self.rew_scale = agent_params['rew_scale']

    def estimate_advantage(self, ob_no, ac_na):
        """Computes A(s, a) = Q(s, a) - V(s) using the V-network.

        Args:
            ob_no: np.ndarray of shape [B, ob_dim].
            ac_na: np.ndarray of shape [B] (discrete: integer action indices)
                   or [B, ac_dim] (continuous: float action vectors).

        Returns:
            FloatTensor of shape [B] containing A(s, a) for each sample.
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        if self.discrete_action:
            ac_na = ac_na.to(torch.long)

        with torch.no_grad():
            v_pi = self.critic.v_net(ob_no).squeeze(-1)
            q_sa = self.critic.get_q(ob_no, ac_na)
            # TODO: Estimate the advantage function.
            ### YOUR CODE START HERE ###
            adv = None
            ### YOUR CODE END HERE ###
        return adv

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        """Performs one training iteration (V-update, actor update, Q-update).

        Args:
            ob_no: np.ndarray of shape [B, ob_dim].
            ac_na: np.ndarray of shape [B] (discrete: integer action indices)
                   or [B, ac_dim] (continuous: float action vectors).
            re_n: np.ndarray of shape [B].
            next_ob_no: np.ndarray of shape [B, ob_dim].
            terminal_n: np.ndarray of shape [B].

        Returns:
            Dict of scalar training metrics.
        """
        log = {}

        if (self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            env_reward = (re_n + self.rew_shift) * self.rew_scale

            critic_loss = self.critic.update_v(ob_no, ac_na)

            # TODO: update actor
            # 1): Estimate the advantage
            # 2): Calculate the awac actor loss
            ### YOUR CODE START HERE ###

            actor_loss = None
            ### YOUR CODE END HERE ###

            critic_loss.update(
                self.critic.update_q(
                    ob_no, ac_na, next_ob_no, env_reward, terminal_n)
            )

            self.critic.update_target_network()

            log['Critic V Loss'] = critic_loss['Training V Loss']
            log['Critic Q Loss'] = critic_loss['Training Q Loss']
            if 'Training Q Loss2' in critic_loss:
                log['Critic Q Loss2'] = critic_loss['Training Q Loss2']
            log['Actor Loss'] = actor_loss
            log['Actor LR'] = self.actor.optimizer.param_groups[0]['lr']

            self.num_param_updates += 1

        self.t += 1
        return log
