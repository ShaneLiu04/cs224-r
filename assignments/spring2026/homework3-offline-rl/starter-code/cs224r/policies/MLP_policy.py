# =============================================================================
# MLP_policy.py  —  MLP policy classes used by AWAC, IQL, and BC
# =============================================================================
# FILE STRUCTURE
# --------------
# MLPPolicy  
#   __init__      : builds either a discrete logits network (logits_na) or a
#                   continuous Gaussian policy (mean_net + logstd parameter),
#                   depending on the `discrete` flag.
#   get_action()  : samples one action from the policy.
#   forward()     : given a batch of observations, returns a torch Distribution
#                   (Categorical for discrete, MultivariateNormal for continuous).
#   update()      : abstract — subclasses must implement.
#
# MLPPolicyAWAC  (extends MLPPolicy)
#   update()      : advantage-weighted regression (AWR) policy update.
#
# MLPPolicyBC    (extends MLPPolicy) 
#   update()      : standard behavior cloning (MLE) loss.
#
# WHERE TO EDIT
# -------------
# * MLPPolicyAWAC.update()  —  implement the AWR loss inside
# =============================================================================

import abc
import itertools

import numpy as np
import torch
from torch import distributions, nn, optim

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for MLP-parameterized policies."""

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim, output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def get_action(self, obs: np.ndarray, deterministic: bool = False,
                   temperature: float = 1.0) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        action_distribution = self(observation, temperature=temperature)
        if deterministic:
            if self.discrete:
                action = action_distribution.logits.argmax(dim=-1)
            else:
                action = action_distribution.mean
        else:
            action = action_distribution.sample()  # [1, ac_dim] or [1]
        action = ptu.to_numpy(action[0])  # remove batch dim
        if not self.discrete:
            action = np.clip(action, -1.0, 1.0)
        return action

    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    def forward(self, observation: torch.FloatTensor, temperature: float = 1.0):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = torch.tanh(self.mean_net(observation))
            clipped_logstd = torch.clamp(self.logstd, min=-5.0, max=2.0)
            std = torch.exp(clipped_logstd) * float(temperature)
            scale_tril = torch.diag(std)
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution


class MLPPolicyAC(MLPPolicy):
    """Actor-critic policy stub (not used in this assignment)."""

    def update(self, observations, actions, adv_n=None, acs_labels_na=None,
               qvals=None):
        raise NotImplementedError


class MLPPolicyAWAC(MLPPolicy):
    """Advantage-Weighted Actor-Critic (AWAC / AWR) policy."""

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 lambda_awac=1,
                 max_grad_norm=1.0,
                 max_steps=None,
                 **kwargs,
                 ):
        self.lambda_awac = lambda_awac
        self.max_grad_norm = max_grad_norm
        super().__init__(ac_dim, ob_dim, n_layers, size, discrete,
                         learning_rate, **kwargs)

        # Cosine LR decay.
        if max_steps is not None and max_steps > 0:
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps
            )
        else:
            self.lr_scheduler = None

    def update(self, observations, actions, adv_n=None):
        """Performs one AWR policy gradient step.

        Args:
            observations: np.ndarray or FloatTensor of shape [B, ob_dim].
            actions: np.ndarray or LongTensor of shape [B] (discrete) or
                FloatTensor of shape [B, ac_dim] (continuous).
            adv_n: np.ndarray or FloatTensor of shape [B], advantage
                estimates. Required.

        Returns:
            Scalar float — the actor loss value.
        """
        assert adv_n is not None, (
            'adv_n (advantage estimates) must be provided for AWAC update')
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)

        dist = self(observations)
        log_prob_n = dist.log_prob(actions)
        # TODO: Use adv_n and self.lambda_awac to compute exponential weights.
        ### YOUR CODE START HERE ###
        exp_weights = None
        ### YOUR CODE END HERE ###
        exp_weights = exp_weights.clamp(max=50.0)
        actor_loss = -(log_prob_n * exp_weights).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return actor_loss.item()


class MLPPolicyBC(MLPPolicy):
    """Behavioral Cloning policy: pure MLE on the offline dataset."""

    def __init__(self, *args, max_grad_norm=1.0, max_steps=None, **kwargs):
        self.max_grad_norm = max_grad_norm
        super().__init__(*args, **kwargs)
        if max_steps is not None and max_steps > 0:
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps
            )
        else:
            self.lr_scheduler = None

    def update(self, observations, actions, **kwargs):
        """Performs one MLE supervised update step.

        Args:
            observations: np.ndarray or FloatTensor of shape [B, ob_dim].
            actions: np.ndarray or LongTensor of shape [B] (discrete) or
                FloatTensor of shape [B, ac_dim] (continuous).
            **kwargs: Ignored.

        Returns:
            Scalar float — the BC loss value.
        """
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)

        dist = self(observations)
        loss = -dist.log_prob(actions).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()
