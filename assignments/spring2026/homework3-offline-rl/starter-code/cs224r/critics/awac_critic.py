# =============================================================================
# awac_critic.py  —  Double Q-function critic for AWAC
# =============================================================================
#
# FILE STRUCTURE
# --------------
# AWACCritic (extends BaseCritic)
#   __init__          : builds two independent Q-networks (q_net, q_net2) each
#                       with a frozen target copy (q_net_target, q_net2_target).
#
#   _get_q_value()    : helper — returns Q(s,a) from a Q-network
# .
#   get_q()           : returns min(Q1, Q2)(s, a) — used for conservative
#                       advantage estimation in the actor update.
#
#   update()          : One gradient step on both Q-networks jointly.
#
#   update_target_network() : soft EMA update for both target pairs.
#
#   qa_values()       : returns Q1(s,·) for all actions (used for action
#                       selection and visualisation).
#
# WHERE TO EDIT
# -------------
# * update()  —  inside the ### YOUR CODE START/END ### block:
#       Compute the clipped double-Q TD target using get_target_q(next_ob_no,
#       next_actions), then compute MSE losses for both q_net and q_net2.
#       See the HINTs in the docstring for guidance.
# =============================================================================

import torch
import torch.optim as optim
from torch import nn
from torch.nn import utils

from cs224r.infrastructure import pytorch_util as ptu
from .base_critic import BaseCritic


class AWACCritic(BaseCritic):
    """Q-function critic for AWAC."""

    def __init__(self, hparams, optimizer_spec, **kwargs):
        """Initializes AWACCritic.

        Args:
            hparams: Dict of hyperparameters with keys:
                ob_dim: Observation dimension.
                ac_dim: Action dimension (number of discrete actions, or
                    continuous action vector size).
                discrete: If the action space is discrete.
                grad_norm_clipping: Max gradient norm.
                gamma: Discount factor.
                q_func: Callable (ob_dim, ac_dim) -> Q-network.
                    Discrete: outputs [B, ac_dim].
                    Continuous: takes (obs, action) and outputs [B, 1].
                awac_tau: EMA coefficient for the soft target update
                    (default 0.005).
            optimizer_spec: OptimizerSpec with constructor, optim_kwargs, and
                learning_rate_schedule.
        """
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete_action = hparams['discrete']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']

        # Double Q-learning
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.q_net2 = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net2_target = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net2.to(ptu.device)
        self.q_net2_target.to(ptu.device)
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())

        q_params = list(self.q_net.parameters()) + list(self.q_net2.parameters())
        self.optimizer = self.optimizer_spec.constructor(
            q_params,
            **self.optimizer_spec.optim_kwargs,
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.mse_loss = nn.MSELoss()
        self.tau = hparams.get('awac_tau', 0.005)  # EMA coefficient for soft target update.

    def _get_q_value(self, q_net, obs, actions):
        if self.discrete_action:
            qa_values = q_net(obs)
            return torch.gather(qa_values, 1, actions.long().unsqueeze(1)).squeeze(1)
        else:
            return q_net(obs, actions).squeeze(-1)

    def get_q(self, obs, actions):
        """Returns min(Q1, Q2)(s, a).

        Args:
            obs: FloatTensor of shape [B, ob_dim].
            actions: LongTensor of shape [B] (discrete) or
                FloatTensor of shape [B, ac_dim] (continuous).

        Returns:
            FloatTensor of shape [B].
        """
        q1 = self._get_q_value(self.q_net, obs, actions)
        q2 = self._get_q_value(self.q_net2, obs, actions)
        return torch.min(q1, q2)

    def get_target_q(self, obs, actions):
        """Returns min(Q1_target, Q2_target)(s, a).

        Args:
            obs: FloatTensor of shape [B, ob_dim].
            actions: LongTensor of shape [B] (discrete) or
                FloatTensor of shape [B, ac_dim] (continuous).

        Returns:
            FloatTensor of shape [B].
        """
        q1 = self._get_q_value(self.q_net_target, obs, actions)
        q2 = self._get_q_value(self.q_net2_target, obs, actions)
        return torch.min(q1, q2)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n,
               next_actions):
        """Performs one gradient step on the online Q-network.

        Computes the double-Q Bellman target and minimises 
            MSE(Q1(s,a), y) + MSE(Q2(s,a), y).

        Args:
            ob_no: np.ndarray of shape [B, ob_dim], current observations.
            ac_na: np.ndarray of shape [B] (discrete: integer action indices)
                   or [B, ac_dim] (continuous: float action vectors).
            next_ob_no: np.ndarray of shape [B, ob_dim], next observations.
            reward_n: np.ndarray of shape [B], rewards.
            terminal_n: np.ndarray of shape [B], done flags (1 = terminal).
            next_actions: LongTensor of shape [B] (discrete) or FloatTensor
                of shape [B, ac_dim] (continuous), one action sampled from
                π(·|s') per transition.

        Returns:
            Dict with 'Training Loss' (Q1 MSE) and 'Training Loss2' (Q2 MSE).
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        if self.discrete_action:
            ac_na = ac_na.to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # TODO: Compute loss for updating Q_net and Q_net2 parameters.
        # HINT: Compute Q1(s,a) and Q2(s,a) using _get_q_value.
        # HINT: Compute the TD target using get_target_q
        # HINT: Stop gradients from flowing into the target networks by using
        #     torch.no_grad() or by detaching the target from the graph.
        # HINT: Note that if the next state is terminal, its target value
        #     needs to be adjusted.
        # HINT: Compute MSE losses loss1 and loss2 for q_net and q_net2.
        ### YOUR CODE START HERE ###

        # TD target = r + γ * min(Q1_target, Q2_target)(s', a') * (1 - done).
        
        loss = None
        loss2 = None
        ### YOUR CODE END HERE ###

        self.optimizer.zero_grad()
        (loss + loss2).backward()
        utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.q_net2.parameters()),
            self.grad_norm_clipping,
        )
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {'Training Loss': ptu.to_numpy(loss), 'Training Loss2': ptu.to_numpy(loss2)}

    def update_target_network(self):
        """Soft EMA update: Q_target ← (1-τ)*Q_target + τ*Q_online."""
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)
        for target_param, param in zip(
                self.q_net2_target.parameters(), self.q_net2.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

    def qa_values(self, obs):
        """Returns Q-values for all actions (used for visualization).

        Args:
            obs: np.ndarray of shape [B, ob_dim].

        Returns:
            np.ndarray of shape [B, ac_dim].
        """
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
