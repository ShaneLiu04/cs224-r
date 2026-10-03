import copy
import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils



class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_init=-2.0):
        """Build the policy network that parameterizes a bounded action distribution."""
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_shape[0], hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.log_std = nn.Linear(hidden_dim, action_shape[0])


        self.apply(utils.weight_init)


    def forward(self, obs):
        """Map observations to a truncated Gaussian action distribution."""
        # Run through all but last layer to get shared features
        obs = obs.float()
        features = self.policy[:-1](obs)
        mu = torch.tanh(self.policy[-1](features))
        std = torch.exp(self.log_std(features)).clamp(1e-5, 1.0)

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, hidden_dim):
        """Build the value function used to estimate state values for PPO."""
        super().__init__()

        # PPO uses a single value head (V(s)), not Q(s,a)
        self.value_net = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs):
        """Predict a scalar value for each observation in the batch."""
        obs = obs.float()
        return self.value_net(obs)


class PPOAgent:
    def __init__(self, obs_shape, action_shape, device, lr, batch_size,
                 hidden_dim, clip_eps, ppo_epochs, value_coef,
                 entropy_coef, gae_lambda, gamma,
                 reverse_kl_coef=1e-3, std=0.1):
        """Construct PPO actor/critic modules and store the optimization hyperparameters."""
        self.device = device
        self.batch_size = batch_size

        # PPO hyperparameters
        self.clip_eps = clip_eps # epsilon for clipped surrogate objective 
        self.ppo_epochs = ppo_epochs # number of update epochs per rollout 
        self.value_coef = value_coef # weight on value loss 
        self.entropy_coef = entropy_coef # weight on entropy bonus 
        self.gae_lambda = gae_lambda # GAE lambda 
        self.gamma = gamma # discount factor
        self.reverse_kl_coef = reverse_kl_coef # weight on reverse KL regularization

        # models
        self.actor = Actor(obs_shape, action_shape, hidden_dim, std).to(device)
        self.critic = Critic(obs_shape, hidden_dim).to(device)
        self.reference_actor = None

        # optimizer
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.train()

    def train(self, training=True):
        """Switch the actor and critic between training and evaluation modes."""
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def set_reference_policy(self):
        """Freeze a reference policy after BC for KL during PPO updates."""
        self.reference_actor = copy.deepcopy(self.actor).to(self.device)
        self.reference_actor.train(False)
        for param in self.reference_actor.parameters():
            param.requires_grad_(False)

    def act(self, obs, eval_mode):
        """Select one action for a single observation, either greedily or by sampling."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def compute_gae(self, rewards, values, next_values, discounts, dones):
        """
        Compute Generalised Advantage Estimates (GAE) and returns.

        Args:
            rewards: rewards
            values: values of current states
            next_values: values of next states
            discounts: discount
            dones: dones

        Returns:
            advantages: computed advantages
            returns: advantages + values, used as V targets
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=self.device)

        for t in reversed(range(T)):

            ### YOUR CODE HERE ###


            ### YOUR CODE HERE ###

        returns = advantages + values
        return advantages, returns

    def update(self, rollout_buffer):
        """
        PPO update: runs ppo_epochs passes over the collected rollout.

        Args:
            rollout_buffer:
                An object (or iterable) that yields batches of tuples
                (obs, action, reward, discount, next_obs, done, old_log_prob)
                collected under the *current* policy before the update.

                obs: shape [batch, D]
                action: shape [batch, action_dim]
                reward: shape [batch]
                discount: shape [batch]   gamma * (1 - done), pre-computed
                next_obs: shape [batch, D]
                done: shape [batch], episode termination flag
                old_log_prob: [batch], log π_old(a|s), recorded during rollout

        Returns:
            metrics: dict of logging values
        """
        metrics = dict()
        if self.reference_actor is None:
            raise RuntimeError(
                "Reference policy not set. Call set_reference_policy() after BC pretraining.")

        # Collect the full rollout
        obs_list, act_list, rew_list, disc_list = [], [], [], []
        next_obs_list, done_list, old_lp_list = [], [], []

        for batch in rollout_buffer:
            obs, action, reward, discount, next_obs, done, old_log_prob = \
                utils.to_torch(batch, self.device)
            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)
            disc_list.append(discount)
            next_obs_list.append(next_obs)
            done_list.append(done)
            old_lp_list.append(old_log_prob)

        obs_all = torch.cat(obs_list, dim=0) # [T, D]
        act_all = torch.cat(act_list, dim=0) # [T, A]
        rew_all = torch.cat(rew_list, dim=0) # [T]
        disc_all = torch.cat(disc_list, dim=0) # [T]
        next_obs_all = torch.cat(next_obs_list, dim=0) # [T, D]
        done_all = torch.cat(done_list, dim=0) # [T]
        old_lp_all = torch.cat(old_lp_list, dim=0) # [T]

        # Compute GAE advantages
        with torch.no_grad():


            ### YOUR CODE HERE ###
            
                        
            ### YOUR CODE HERE ###


            adv_std = advantages_all.std(unbiased=False)

            advantages_all_old = advantages_all

            advantages_all = (advantages_all - advantages_all.mean()) / (adv_std + 1e-8)

        T = obs_all.shape[0]

        # PPO epoch loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_reverse_kl = 0.0
        num_updates = 0

        params = list(self.actor.parameters()) + list(self.critic.parameters())

        for _ in range(self.ppo_epochs):
            # Shuffle indices each epoch
            idx = torch.randperm(T, device=self.device)
            self.opt.zero_grad(set_to_none=True)

            for i in range(0, T, self.batch_size):
                batch_idx = idx[i:i + self.batch_size]
                batch_size = batch_idx.shape[0]

                obs_ep = obs_all[batch_idx]
                act_ep = act_all[batch_idx]
                adv_ep = advantages_all[batch_idx]
                ret_ep = returns_all[batch_idx]
                olp_ep = old_lp_all[batch_idx]

                # New log-probs under current policy
                dist = self.actor(obs_ep)
                new_log_prob = dist.log_prob(act_ep).sum(-1)   # [batch_size]
                entropy = dist.entropy().sum(-1).mean()   # scalar

                # Clipped surrogate (PPO-Clip objective) 

                ### YOUR CODE HERE ###

                
                
                

                ### YOUR CODE HERE ###

                # Reverse KL to frozen reference policy:
                # D_KL(pi_new || pi_ref) = E_old[ratio * (log pi_new - log pi_ref)]
                with torch.no_grad():
                    ref_dist = self.reference_actor(obs_ep)
                    ref_log_prob = ref_dist.log_prob(act_ep).sum(-1)
                reverse_kl   = (ratio * (new_log_prob - ref_log_prob)).mean()

                # Value loss
                values_pred = self.critic(obs_ep).squeeze(-1)
                value_loss = F.mse_loss(values_pred, ret_ep)

                # Combined loss 
                loss = (policy_loss
                        + self.value_coef  * value_loss
                        - self.entropy_coef * entropy
                        + self.reverse_kl_coef * reverse_kl)  # minus: we maximise entropy

                # Accumulate gradients over minibatches, then step once per epoch.
                (loss * (batch_size / T)).backward()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_reverse_kl += reverse_kl.item()
                num_updates += 1

            nn.utils.clip_grad_norm_(params, max_norm=1)
            self.opt.step()

        # Logging
        if num_updates > 0:
            metrics['policy_loss']  = total_policy_loss / num_updates
            metrics['value_loss']   = total_value_loss  / num_updates
            metrics['entropy']      = total_entropy     / num_updates
            metrics['reverse_kl']   = total_reverse_kl  / num_updates
            metrics['batch_reward'] = rew_all.mean().item()
            metrics['advantage_mean'] = advantages_all.mean().item()
            metrics['raw_advantage_mean'] = advantages_all_old.mean().item()
            metrics['raw_advantage_std'] = advantages_all_old.std(unbiased=False).item()
            metrics['advantage_std'] = advantages_all.std(unbiased=False).item()
            metrics['returns_mean'] = returns_all.mean().item()
            metrics['returns_std'] = returns_all.std(unbiased=False).item()

            with torch.no_grad():
                dist_all = self.actor(obs_all)
                metrics['actor_std'] = dist_all.scale.mean().item()

        return metrics

    def bc(self, replay_iter):
        """
        Behaviour cloning pre-training

        Args:
            replay_iter: iterable of (obs, action, reward, discount, next_obs)

        Returns:
            metrics: dict
        """
        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        dist = self.actor(obs)
        actor_loss = -dist.log_prob(action).sum(-1, keepdim=True).mean()

        self.opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt.step()

        metrics['pretrain_actor_loss'] = actor_loss.item()
        metrics['actor_std'] = dist.scale.mean().item()

        return metrics
