# =============================================================================
# iql_critic.py  —  Double Q-function and V-function critics for IQL
# =============================================================================
# FILE STRUCTURE
# --------------
# IQLCritic (extends BaseCritic)
#   __init__          : builds two independent Q-networks (q_net, q_net2) each
#                       with a frozen target copy (q_net_target, q_net2_target),
#                       and v_net (MLP, output dim = 1).
#
#   _get_q_value()    : helper — for discrete, gathers Q(s,a) from full Q output
#                       [B, ac_dim] using action indices → [B]; for continuous,
#                       calls q_net(obs, actions) directly → [B].
#   get_q()           : returns min(Q1, Q2)(s, a).
#
#   expectile_loss()  : Computes the asymmetric expectile loss.
#
#   update_v()        : One gradient step on v_net using the expectile loss.
#
#   update_q()        : One gradient step on both Q-networks jointly.
#
#   update_target_network() : soft EMA update for both target pairs.
#
#   qa_values()       : returns Q1(s,·) for all actions (used for action
#                       selection and visualisation; discrete only).
#
# WHERE TO EDIT
# -------------
# * __init__()        —  define self.v_net inside ### YOUR CODE START/END ###.
#
# * expectile_loss()  —  implement the asymmetric expectile loss inside 
#                         ### YOUR CODE START/END HERE ###.
#
# * update_v()        —  compute the V-function loss using expectile_loss() and
#                         get_target_q() inside ### YOUR CODE START/END ###.
#
# * update_q()        —  compute the shared Bellman target using v_net(s'), then
#                         compute and sum MSE losses for both Q-networks inside
#                         ### YOUR CODE START/END ###.
# =============================================================================

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import utils

from cs224r.infrastructure import pytorch_util as ptu
from .base_critic import BaseCritic


class IQLCritic(BaseCritic):
    """Q-function and V-function critics for IQL (discrete and continuous actions)."""

    def __init__(self, hparams, optimizer_spec, **kwargs):
        """Initializes IQLCritic.

        Args:
            hparams: Dict of hyperparameters with keys:
                ob_dim: Observation dimension.
                ac_dim: Action dimension (number of discrete actions, or
                    continuous action vector size).
                discrete: True for discrete action spaces, False for continuous.
                grad_norm_clipping: Max gradient norm.
                gamma: Discount factor.
                q_func: Callable (ob_dim, ac_dim) -> Q-network.
                    Discrete: outputs logits [B, ac_dim].
                    Continuous: takes (obs, action) and outputs [B, 1].
                v_func: Callable (ob_dim) -> V-network with output shape [B, 1].
                iql_expectile: Expectile ζ for the asymmetric loss.
                iql_tau: EMA coefficient for the soft target update
                    (default 0.005).
            optimizer_spec: OptimizerSpec with constructor, optim_kwargs, and
                learning_rate_schedule.
            **kwargs: Passed to BaseCritic.
        """
        # kwargs are passed through from get_env_kwargs in utils.py
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams.get('discrete', True)
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        q_network_initializer = hparams['q_func']
        v_network_initializer = hparams['v_func']

        self.q_net = q_network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = q_network_initializer(self.ob_dim, self.ac_dim)
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.q_net2 = q_network_initializer(self.ob_dim, self.ac_dim)
        self.q_net2_target = q_network_initializer(self.ob_dim, self.ac_dim)
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

        # TODO: define value function.
        # HINT: Use v_network_initializer.
        # HINT: See the Q_net setup above and v_optimizer setup below for
        #     the pattern to follow.
        ### YOUR CODE START HERE ###


        ### YOUR CODE END HERE ###

        self.v_optimizer = self.optimizer_spec.constructor(
            self.v_net.parameters(),
            **self.optimizer_spec.optim_kwargs,
        )
        self.v_learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.v_optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.iql_expectile = hparams['iql_expectile']
        self.tau = hparams.get('iql_tau', 0.005)  # EMA coefficient for soft target update.

    def _get_q_value(self, q_net, obs, actions):
        """Returns Q(s, a) gathered from the full Q-network output (discrete) or
        squeezed from the network output [B, 1] to [B] (continuous).
        
        Args:
            q_net: The Q-network module.
            obs: FloatTensor of shape [B, ob_dim].
            actions: LongTensor of shape [B] (discrete) or FloatTensor [B, ac_dim] (continuous).

        Returns:
            FloatTensor of shape [B] containing Q(s, a).
        """
        if self.discrete:
            qa_values = q_net(obs)          # [B, ac_dim]
            return torch.gather(qa_values, 1, actions.long().unsqueeze(1)).squeeze(1)
        else:
            return q_net(obs, actions).squeeze(-1)

    def get_q(self, obs, actions):
        """Returns min(Q1, Q2)(s, a) for conservative advantage estimation.

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

    def expectile_loss(self, diff):
        """Computes the asymmetric expectile loss L_ζ(δ) = |ζ - 𝟙[δ<0]| · δ².

        Args:
            diff: FloatTensor of shape [B], the difference Q - V.

        Returns:
            FloatTensor of shape [B], per-sample expectile loss.
        """
        # TODO: Implement the expectile loss given the difference between q
        #     and v.
        # HINT: self.iql_expectile provides the ζ value as described in the
        #     problem statement.
        ### YOUR CODE START HERE ###

        pass
        ### YOUR CODE END HERE ###

    def update_v(self, ob_no, ac_na):
        """Updates the value function using the expectile loss.

        Args:
            ob_no: np.ndarray of shape [B, ob_dim].
            ac_na: np.ndarray of shape [B] with integer action indices
                (discrete) or shape [B, ac_dim] with float action vectors
                (continuous). Used to compute the Q target for the expectile
                regression.

        Returns:
            Dict mapping 'Training V Loss' to a scalar np.ndarray.
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        if self.discrete:
            ac_na = ac_na.to(torch.long)

        # Compupte the target Q-values
        with torch.no_grad():
            q_t_values = self.get_target_q(ob_no, ac_na)

        # Compute V(s) using the value network.
        v_t = self.v_net(ob_no).squeeze(-1)
        assert q_t_values.shape == v_t.shape

        # TODO: Compute loss for v_net.
        # HINT: Apply expectile regression between the target Q-values and the V-values to get the V-network loss.
        ### YOUR CODE START HERE ###
        value_loss = None
        ### YOUR CODE END HERE ###

        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_norm_(self.v_net.parameters(), self.grad_norm_clipping)
        self.v_optimizer.step()
        self.v_learning_rate_scheduler.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}

    def update_q(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """Updates the Q-network using the IQL Bellman target y = r + γV(s').

        Args:
            ob_no: np.ndarray of shape [B, ob_dim].
            ac_na: np.ndarray of shape [B] with integer action indices
                (discrete) or shape [B, ac_dim] with float action vectors
                (continuous).
            next_ob_no: np.ndarray of shape [B, ob_dim].
            reward_n: np.ndarray of shape [B].
            terminal_n: np.ndarray of shape [B], done flags.

        Returns:
            Dict mapping 'Training Q Loss' and 'Training Q Loss2' to scalar
            np.ndarrays (losses for q_net and q_net2 respectively).
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        if self.discrete:
            ac_na = ac_na.to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # TODO: Compute loss for updating Q_net and Q_net2 parameters.
        # HINT: Compute the TD target using v_net.
        # HINT: Stop gradients from flowing into the target networks by using
        #     torch.no_grad() or by detaching the target from the graph.
        # HINT: Note that if the next state is terminal, its target value
        #     needs to be adjusted.
        # HINT: Compute MSE losses for both q_net and q_net2.
        ### YOUR CODE START HERE ###
        
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

        return {'Training Q Loss': ptu.to_numpy(loss), 'Training Q Loss2': ptu.to_numpy(loss2)}

    def update_target_network(self):
        """Soft EMA update: Q_target ← (1-τ)*Q_target + τ*Q."""
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)
        for target_param, param in zip(
                self.q_net2_target.parameters(), self.q_net2.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau).add_(param.data * self.tau)

    def qa_values(self, obs):
        """Returns Q-values for all actions (used for density visualization).

        Args:
            obs: np.ndarray of shape [B, ob_dim].

        Returns:
            np.ndarray of shape [B, ac_dim].
        """
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
