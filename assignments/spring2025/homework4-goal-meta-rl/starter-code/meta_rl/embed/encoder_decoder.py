"""Defines neural network components for encoding from ids and decoding from trajectories."""

import abc
import collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embed.embedders import *
from infrastructure import relabel, rl_utils


class EncoderDecoder(Embedder, relabel.RewardLabeler):
    """Represents both the decoder q(z | \tauexp) and encoder F(z | id).

    When use_ids is set to True, returns the encoders encoding z ~ F(z | id).
    Otherwise, returns the decoding q(z | \tauexp).
    """

    def __init__(self, transition_embedder, id_embedder, penalty, embed_dim):
        """Constructs.

        Args:
            transition_embedder (TransitionEmbedder): embeds a single
                (s, a, s') tuple.
            id_embedder (IDEmbedder): embeds the environment ID as z = f(id)
            penalty (float): per-timestep reward penalty c.
            embed_dim (int): dimension of z.
        """
        super().__init__(embed_dim)

        self._transition_embedder = transition_embedder
        self._id_embedder = id_embedder
        self._transition_lstm = nn.LSTM(transition_embedder.embed_dim, 128)
        self._transition_fc_layer = nn.Linear(128, 128)
        self._transition_output_layer = nn.Linear(128, embed_dim)
        self._penalty = penalty
        self._use_ids = True

    def use_ids(self, use_ids):
        """Controls whether to use the encoder F or the decoder q."""
        self._use_ids = use_ids

    def _compute_embeddings(self, trajectories):
        """Returns embeddings and masks.

        Args:
            trajectories (list[list[Experience]]): see forward().

        Returns:
            id_embeddings (torch.FloatTensor): tensor of shape (batch_size,
                embed_dim) embedding the id's in the trajectories.
                Represents z ~ F_psi(mu).
            all_decoder_embeddings (torch.FloatTensor): tensor of shape
                (batch_size, episode_length + 1, embed_dim) embedding the
                sequences of states and actions in the trajectories:
                index [batch, t] represents g_omega(tau^{exp}_{:t}).
            decoder_embeddings (torch.FloatTensor): tensor of shape
                (batch_size, embed_dim) equal to the last unpadded value in
                all_decoder_embeddings.
            mask (torch.BoolTensor): tensor of shape
                (batch_size, episode_length + 1). The value is False if the
                decoder_embeddings value should be masked.
        """
        # trajectories: (batch_size, max_len)
        # mask: (batch_size, max_len)
        padded_trajectories, mask = rl_utils.pad(trajectories)
        sequence_lengths = torch.tensor(
                [len(traj) for traj in trajectories]).long()

        # (batch_size * max_len, embed_dim)
        transition_embed = self._transition_embedder(
                [exp for traj in padded_trajectories for exp in traj])

        # pack_padded_sequence relies on the default tensor type not
        # being a CUDA tensor.
        torch.set_default_tensor_type(torch.FloatTensor)
        # Sorted only required for ONNX
        padded_transitions = nn.utils.rnn.pack_padded_sequence(
                transition_embed.reshape(mask.shape[0], mask.shape[1], -1),
                sequence_lengths, batch_first=True, enforce_sorted=False)
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        transition_hidden_states = self._transition_lstm(padded_transitions)[0]
        # (batch_size, max_len, hidden_dim)
        transition_hidden_states, hidden_lengths = (
                nn.utils.rnn.pad_packed_sequence(
                    transition_hidden_states, batch_first=True))
        initial_hidden_states = torch.zeros(
                transition_hidden_states.shape[0], 1,
                transition_hidden_states.shape[-1])
        # (batch_size, max_len + 1, hidden_dim)
        transition_hidden_states = torch.cat(
                (initial_hidden_states, transition_hidden_states), 1)
        transition_hidden_states = F.relu(
                self._transition_fc_layer(transition_hidden_states))
        # (batch_size, max_len + 1, embed_dim)
        all_decoder_embeddings = self._transition_output_layer(
                transition_hidden_states)

        # (batch_size, 1, embed_dim)
        # Don't need to subtract 1 off of hidden_lengths as decoder_embeddings
        # is padded with init hidden state at the beginning.
        indices = hidden_lengths.unsqueeze(-1).unsqueeze(-1).expand(
                hidden_lengths.shape[0], 1,
                all_decoder_embeddings.shape[2]).to(
                        all_decoder_embeddings.device)
        decoder_embeddings = all_decoder_embeddings.gather(
                1, indices).squeeze(1)

        # (batch_size, embed_dim)
        id_embeddings = self._id_embedder(
                torch.tensor([traj[0].state.env_id for traj in trajectories]))

        # don't mask the initial hidden states (batch_size, max_len + 1)
        mask = torch.cat(
                (torch.ones(decoder_embeddings.shape[0], 1).bool(), mask), -1)
        return id_embeddings, all_decoder_embeddings, decoder_embeddings, mask

    def _compute_losses(
            self, 
            trajectories, 
            id_embeddings, 
            all_decoder_embeddings,
            decoder_embeddings, 
            mask):
        # pylint: disable=unused-argument
        """Computes losses based on the return values of _compute_embeddings.

        Args:
            See return values of _compute_embeddings.

        Returns:
            losses (dict(str: torch.FloatTensor)): see forward().
        """
        # Unused variables
        del trajectories
        del decoder_embeddings
        
        decoder_context_loss = None
        
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        # TODO: Compute decoder_context_loss, representing the loss function
        # which will be differentiated with respect to the decoder parameters
        # omega.
        #
        # This should be a tensor of shape (batch_size, episode_len + 1)
        # where
        #
        #   decoder_context_loss[batch][t] = ||g_omega(tau_{:t}) - z||^2_2
        #
        # Hint 1: all_decoder_embeddings is a tensor of size (batch_size, episode_len+1, embed_dim) 
        # such that all_decoder_embeddings[batch][t] represents g_omega(tau_{:t}).
        # Hint 2: id_embeddings is a tensor of size (batch_size, embed_dim) such that 
        # id_embeddings[batch] represents z for that batch
        # Hint 3: Reminder that we want to use stop_gradient(z). The torch.tensor.detach 
        # function may be helpful.
        #
        # You may not need to use all of the parameters passed to this function.
        #
        # Parts of the decoder_context_loss are masked below for you to handle
        # batches of episodes of different lengths. You do not need to do
        # anything about this.
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

        decoder_context_loss = (
                decoder_context_loss * mask).sum() / mask.sum()

        cutoff = torch.ones(id_embeddings.shape[0]) * 10
        losses = {
            "decoder_loss": decoder_context_loss,
            "information_bottleneck": torch.max(
                (id_embeddings ** 2).sum(-1), cutoff).mean()
        }
        return losses

    def forward(self, trajectories):
        """Embeds a batch of trajectories to produce z and computes the
        information bottleneck and decoder losses.

        Args:
            trajectories (list[list[Experience]]): batch of trajectories, where
                each trajectory comes from the same episode.

        Returns:
            trajectory_embeddings (torch.FloatTensor): tensor of shape (batch_size,
                    embed_dim) embedding the trajectories. This embedding is based
                    on the ids if use_ids is True, otherwise based on the
                    transitions.
            losses (dict(str: torch.FloatTensor)): maps auxiliary loss names to
                their values.
        """
        id_embeddings, all_decoder_embeddings, decoder_embeddings, mask = (
                self._compute_embeddings(trajectories))
        trajectory_embeddings = (id_embeddings + 0.1 * torch.randn_like(id_embeddings)
                    if self._use_ids else decoder_embeddings)

        losses = self._compute_losses(
                trajectories, id_embeddings, all_decoder_embeddings,
                decoder_embeddings, mask)
        return trajectory_embeddings, losses

    def label_rewards(self, trajectories):
        """Computes rewards for each experience in the trajectory.

        Args:
            trajectories (list[list[Experience]]): batch of trajectories.

        Returns:
            rewards (torch.FloatTensor): of shape (batch_size, episode_len)
                where rewards[i][j] is the rewards for the experience
                trajectories[i][j].  This is padded with zeros and is detached
                from the graph.
            distances (torch.FloatTensor): of shape (batch_size, episode_len +
                1) equal to ||stop_gradient(z) - g(\tau^e_{:t})||_2^2 for each t.
        """
        # pylint: disable=unused-variable
        id_embeddings, all_decoder_embeddings, _, mask = self._compute_embeddings(
                trajectories)
        # pylint: enable=unused-variable


        distances = None
        rewards = None
        
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        # TODO: Compute rewards and distances.
        # The rewards variable should be of shape (batch_size, episode_len),
        # where
        #   rewards[batch][t] = log q_omega(z | tau_{:t + 1}) -
        #                       log q_omega(z | tau_{:t}).
        #
        # `distances` should be of shape (batch_size, episode_len + 1)
        # where
        #   distances[batch][t] = -log q_omega(z | tau_{:t})
        #
        # Note that tau_{:t} at t = 0 is just s_t, though this is handled for
        # you.
        #
        # Hint 1: all_decoder_embeddings is a tensor of size (batch_size, episode_len+1, embed_dim) 
        # such that all_decoder_embeddings[batch][t] represents g_omega(tau_{:t}).
        # Hint 2: id_embeddings is a tensor of size (batch_size, embed_dim) such that 
        # id_embeddings[batch] represents z for that batch
        # Hint 3: Remember that since we parametrize q_omega(z | tau{:t}) as a gaussian, 
        # we have that log q_omega(z | tau_{:t}) = ||stop_gradient(z) - g(\tau^e_{:t})||_2^2 + C
        # such that C is constants independent of our weights we backprop on.
        # Hint 4: Reminder that we want to use stop_gradient(z). The torch.tensor.detach 
        # function may be helpful.
        #
        # The rewards are subsequently masked, to handle batches of episodes
        # with different lengths, but this is done for you, and you should not
        # have to handle masking.
        #
        # Additionally, a penalty c is applied to the rewards. This is
        # alredy done for you below, and you don't need to do anything.
        # See Equation (5) of the DREAM paper if you're curious.
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        return ((rewards - self._penalty) * mask[:, 1:]).detach(), distances
    

class ExploitationPolicyEmbedder(Embedder):
    """Embeds (s, i, \tau^e) where:

        - s is the current state
        - i is the current instruction / goal
        - \tau^e is an exploration trajectory (s_0, a_0, s_1, ..., s_T)
    """

    def __init__(self, encoder_decoder, obs_embedder, instruction_embedder,
                 embed_dim):
        """Constructs around embedders for each component.

        Args:
            encoder_decoder (EncoderDecoder): embeds batches of \tau^e
                (list[list[rl.Experience]]).
            obs_embedder (Embedder): embeds batches of states s.
            instruction_embedder (Embedder): embeds batches of instructions i.
            embed_dim (int): see Embedder.
        """
        super().__init__(embed_dim)

        self._obs_embedder = obs_embedder
        self._instruction_embedder = instruction_embedder
        self._encoder_decoder = encoder_decoder
        self._fc_layer = nn.Linear(
            obs_embedder.embed_dim + self._encoder_decoder.embed_dim, 256)
        self._final_layer = nn.Linear(256, embed_dim)

    def forward(self, states, hidden_state):
        obs_embed, hidden_state = self._obs_embedder(states, hidden_state)
        trajectory_embed, _ = self._encoder_decoder(
                [state[0].trajectory for state in states])

        if len(obs_embed.shape) > 2:
            trajectory_embed = trajectory_embed.unsqueeze(1).expand(
                    -1, obs_embed.shape[1], -1)

        hidden = F.relu(self._fc_layer(
                torch.cat((obs_embed, trajectory_embed), -1)))
        return self._final_layer(hidden), hidden_state

    def aux_loss(self, experiences):
        _, aux_losses = self._encoder_decoder(
                [exp[0].state.trajectory for exp in experiences])
        return aux_losses

    @classmethod
    def from_config(cls, config, env):
        """Returns a configured ExploitationPolicyEmbedder.

        Args:
            config (Config): see Embedder.from_config.
            env (gym.Wrapper): the environment to run on. Expects this to be
                wrapped with an InstructionWrapper.

        Returns:
            ExploitationPolicyEmbedder: configured according to config.
        """
        obs_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                config.get("obs_embedder").get("embed_dim"))
        # Use SimpleGridEmbeder since these are just discrete vars
        instruction_embedder = SimpleGridStateEmbedder(
                env.observation_space["instructions"],
                config.get("instruction_embedder").get("embed_dim"))
        # Exploitation recurrence is not observing the rewards
        exp_embedder = ExperienceEmbedder(
                obs_embedder, instruction_embedder, None, None, None,
                obs_embedder.embed_dim)
        obs_embedder = RecurrentStateEmbedder(
                exp_embedder, obs_embedder.embed_dim)

        transition_config = config.get("transition_embedder")
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                transition_config.get("state_embed_dim"))
        action_embedder = FixedVocabEmbedder(
                env.action_space.n, transition_config.get("action_embed_dim"))
        reward_embedder = None
        if transition_config.get("reward_embed_dim") is not None:
            reward_embedder = LinearEmbedder(
                    1, transition_config.get("reward_embed_dim"))
        transition_embedder = TransitionEmbedder(
                state_embedder, action_embedder, reward_embedder,
                transition_config.get("embed_dim"))
        id_embedder = IDEmbedder(
                env.observation_space["env_id"].high,
                config.get("transition_embedder").get("embed_dim"))
        if config.get("trajectory_embedder").get("type") == "ours":
            encoder_decoder = EncoderDecoder(
                    transition_embedder, id_embedder,
                    config.get("trajectory_embedder").get("penalty"),
                    transition_embedder.embed_dim)
        else:
            raise ValueError(
                    "Unsupported trajectory embedder "
                    f"{config.get('trajectory_embedder')}")
        return cls(encoder_decoder, obs_embedder, instruction_embedder,
                   config.get("embed_dim"))