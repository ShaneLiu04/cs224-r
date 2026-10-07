"""Behavioral Cloning agent."""

from cs224r.infrastructure.utils import MemoryOptimizedReplayBuffer
from cs224r.policies.MLP_policy import MLPPolicyBC
from .base_agent import BaseAgent


class BCAgent(BaseAgent):
    """Behavioral Cloning agent."""

    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__(env, agent_params)

        self.discrete_action = agent_params.get('discrete', True)
        ac_dim = agent_params['ac_dim']

        buf_size = agent_params.get('replay_buffer_size', 1000000)
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            buf_size, 1, float_obs=True,
            continuous_action=not self.discrete_action, ac_dim=ac_dim
        )

        self.actor = MLPPolicyBC(
            agent_params['ac_dim'],
            agent_params['ob_dim'],
            agent_params['n_layers'],
            agent_params['size'],
            agent_params['discrete'],
            agent_params['learning_rate'],
            max_grad_norm=agent_params.get('grad_norm_clipping', 1.0),
            max_steps=agent_params.get('num_timesteps', None),
        )
        self.eval_policy = self.actor

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        """Performs one supervised update step on the BC policy.

        Args:
            ob_no: np.ndarray of shape [B, ob_dim], current observations.
            ac_na: np.ndarray of shape [B] (discrete: integer action indices)
                   or [B, ac_dim] (continuous: float action vectors).
            re_n: unused by BC.
            next_ob_no: unused by BC.
            terminal_n: unused by BC.

        Returns:
            Dict mapping 'Actor Loss' to a scalar, or empty dict if not
            updating this step.
        """
        log = {}
        if (self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)):
            actor_loss = self.actor.update(ob_no, ac_na)
            log['Actor Loss'] = actor_loss
            self.num_param_updates += 1

        self.t += 1
        return log
