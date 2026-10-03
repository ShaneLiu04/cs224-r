"""Defines neural network embedding components."""
import abc
import collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from envs import grid


class Embedder(abc.ABC, nn.Module):
    """Defines the embedding of an object in the forward method.

    Subclasses should register to the from_config method.
    """

    def __init__(self, embed_dim):
        """Sets the embed dim.

        Args:
            embed_dim (int): the dimension of the outputted embedding.
        """
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        """Returns the dimension of the output (int)."""
        return self._embed_dim

    @classmethod
    def from_config(cls, config):
        """Constructs and returns Embedder from config.

        Args:
            config (Config): parameters for constructing the Embedder.

        Returns:
            Embedder
        """
        config_type = config.get("type")
        if config_type == "simple_grid_state":
            return SimpleGridStateEmbedder.from_config(config)
        elif config_type == "fixed_vocab":
            return FixedVocabEmbedder.from_config(config)
        elif config_type == "linear":
            return LinearEmbedder.from_config(config)
        else:
            raise ValueError(f"Config type {config_type} not supported")


def get_state_embedder(env):
    """Returns the appropriate type of embedder given the environment type."""
    env = env.unwrapped
    if isinstance(env.unwrapped, grid.GridEnv):
        return SimpleGridStateEmbedder
    raise ValueError()


class TransitionEmbedder(Embedder):
    """Embeds tuples of (s, a, r, s')."""

    def __init__(self, state_embedder, action_embedder, reward_embedder,
                 embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._action_embedder = action_embedder
        self._reward_embedder = reward_embedder
        reward_embed_dim = (
                0 if reward_embedder is None else reward_embedder.embed_dim)

        self._transition_embedder = nn.Sequential(
                nn.Linear(
                    self._state_embedder.embed_dim * 2 +
                    self._action_embedder.embed_dim + reward_embed_dim,
                    128),
                nn.ReLU(),
                nn.Linear(128, embed_dim)
        )

    def forward(self, experiences):
        state_embeds = self._state_embedder(
                [exp.state.observation for exp in experiences])
        next_state_embeds = self._state_embedder(
                [exp.next_state.observation for exp in experiences])
        action_embeds = self._action_embedder(
                [exp.action for exp in experiences])
        embeddings = [state_embeds, next_state_embeds, action_embeds]
        if self._reward_embedder is not None:
            embeddings.append(self._reward_embedder(
                    [exp.next_state.prev_reward for exp in experiences]))
        transition_embeds = self._transition_embedder(torch.cat(embeddings, -1))
        return transition_embeds

    @classmethod
    def from_config(cls, config, env):
        raise NotImplementedError()


class RecurrentStateEmbedder(Embedder):
    """Applies an LSTM on top of a state embedding."""

    def __init__(self, state_embedder, embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._lstm_cell = nn.LSTMCell(state_embedder.embed_dim, embed_dim)

    def forward(self, states, hidden_state=None):
        """Embeds a batch of sequences of contiguous states.

        Args:
            states (list[list[np.array]]): of shape
                (batch_size, sequence_length, state_dim).
            hidden_state (list[object] | None): batch of initial hidden states
                to use with the LSTM. During inference, this should just be the
                previously returned hidden state.

        Returns:
            embedding (torch.tensor): shape (batch_size, sequence_length,
                embed_dim)
            hidden_state (object): hidden state after embedding every element
                in the sequence.
        """
        batch_size = len(states)
        sequence_len = len(states[0])

        # Stack batched hidden state
        if batch_size > 1 and hidden_state is not None:
            hs = []
            cs = []
            for hidden in hidden_state:
                if hidden is None:
                    hs.append(torch.zeros(1, self.embed_dim))
                    cs.append(torch.zeros(1, self.embed_dim))
                else:
                    hs.append(hidden[0])
                    cs.append(hidden[1])
            hidden_state = (torch.cat(hs, 0), torch.cat(cs, 0))

        flattened = [state for seq in states for state in seq]

        # (batch_size * sequence_len, embed_dim)
        state_embeds = self._state_embedder(flattened)
        state_embeds = state_embeds.reshape(batch_size, sequence_len, -1)

        embeddings = []
        for seq_index in range(sequence_len):
            hidden_state = self._lstm_cell(
                    state_embeds[:, seq_index, :], hidden_state)

            # (batch_size, 1, embed_dim)
            embeddings.append(hidden_state[0].unsqueeze(1))

        # (batch_size, sequence_len, embed_dim)
        # squeezed to (batch_size, embed_dim) if sequence_len == 1
        embeddings = torch.cat(embeddings, 1).squeeze(1)

        # Detach to save GPU memory.
        detached_hidden_state = (
                hidden_state[0].detach(), hidden_state[1].detach())
        return embeddings, detached_hidden_state

    @classmethod
    def from_config(cls, config, env):
        experience_embed_config = config.get("experience_embedder")
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                experience_embed_config.get("state_embed_dim"))
        action_embedder = FixedVocabEmbedder(
                env.action_space.n + 1,
                experience_embed_config.get("action_embed_dim"))
        instruction_embedder = None
        if experience_embed_config.get("instruction_embed_dim") is not None:
            # Use SimpleGridEmbedder since these are just discrete vars
            instruction_embedder = SimpleGridStateEmbedder(
                    env.observation_space["instructions"],
                    experience_embed_config.get("instruction_embed_dim"))

        reward_embedder = None
        if experience_embed_config.get("reward_embed_dim") is not None:
            reward_embedder = LinearEmbedder(
                    1, experience_embed_config.get("reward_embed_dim"))

        done_embedder = None
        if experience_embed_config.get("done_embed_dim") is not None:
            done_embedder = FixedVocabEmbedder(
                    2, experience_embed_config.get("done_embed_dim"))

        experience_embedder = ExperienceEmbedder(
                state_embedder, instruction_embedder, action_embedder,
                reward_embedder, done_embedder,
                experience_embed_config.get("embed_dim"))
        return cls(experience_embedder, config.get("embed_dim"))


class StateInstructionEmbedder(Embedder):
    """Embeds instructions and states and applies a linear layer on top."""

    def __init__(self, state_embedder, instruction_embedder, embed_dim):
        super().__init__(embed_dim)
        self._state_embedder = state_embedder
        self._instruction_embedder = instruction_embedder
        if instruction_embedder is not None:
            self._final_layer = nn.Linear(
                    state_embedder.embed_dim + instruction_embedder.embed_dim,
                    embed_dim)
            assert self._state_embedder.embed_dim == embed_dim

    def forward(self, states):
        state_embeds = self._state_embedder(
                [state.observation for state in states])
        if self._instruction_embedder is not None:
            instruction_embeds = self._instruction_embedder(
                    [torch.tensor(state.instructions) for state in states])
            return self._final_layer(
                    F.relu(torch.cat((state_embeds, instruction_embeds), -1)))
        return state_embeds


class SimpleGridStateEmbedder(Embedder):
    """Embedder for SimpleGridEnv states.

    Concretely, embeds (x, y) separately with different embeddings for each
    cell.
    """

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (spaces.Box): limits for the observations to
                embed.
        """
        super().__init__(embed_dim)

        assert all(dim == 0 for dim in observation_space.low)
        assert observation_space.dtype == int

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim, hidden_size)
                 for dim in observation_space.high])
        self._fc_layer = nn.Linear(
                hidden_size * len(observation_space.high), 256)
        self._final_fc_layer = nn.Linear(256, embed_dim)

    def forward(self, obs):
        tensor = torch.stack(obs)
        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        return self._final_fc_layer(
                F.relu(self._fc_layer(torch.cat(embeds, -1))))


class IDEmbedder(Embedder):
    """Embeds N-dim IDs by embedding each component and applying a linear
    layer."""

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (np.array): discrete max limits for each
                dimension of the state (expects min is 0).
        """
        super().__init__(embed_dim)

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim, hidden_size) for dim in observation_space])
        self._fc_layer = nn.Linear(
                hidden_size * len(observation_space), embed_dim)

    @classmethod
    def from_config(cls, config, observation_space):
        return cls(observation_space, config.get("embed_dim"))

    def forward(self, obs):
        tensor = obs
        if len(tensor.shape) == 1:  # 1-d IDs
            tensor = tensor.unsqueeze(-1)

        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        return self._fc_layer(torch.cat(embeds, -1))


class FixedVocabEmbedder(Embedder):
    """Wrapper around nn.Embedding obeying the Embedder interface."""

    def __init__(self, vocab_size, embed_dim):
        """Constructs.

        Args:
            vocab_size (int): number of unique embeddings.
            embed_dim (int): dimension of output embedding.
        """
        super().__init__(embed_dim)

        self._embedder = nn.Embedding(vocab_size, embed_dim)

    @classmethod
    def from_config(cls, config):
        return cls(config.get("vocab_size"), config.get("embed_dim"))

    def forward(self, inputs):
        """Embeds inputs according to the underlying nn.Embedding.

        Args:
            inputs (list[int]): list of inputs of length batch.

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        tensor_inputs = torch.tensor(np.stack(inputs)).long()
        return self._embedder(tensor_inputs)


class LinearEmbedder(Embedder):
    """Wrapper around nn.Linear obeying the Embedder interface."""

    def __init__(self, input_dim, embed_dim):
        """Wraps a nn.Linear(input_dim, embed_dim).

        Args:
            input_dim (int): dimension of inputs to embed.
            embed_dim (int): dimension of output embedding.
        """
        super().__init__(embed_dim)

        self._embedder = nn.Linear(input_dim, embed_dim)

    @classmethod
    def from_config(cls, config):
        return cls(config.get("input_dim"), config.get("embed_dim"))

    def forward(self, inputs):
        """Embeds inputs according to the underlying nn.Linear.

        Args:
            inputs (list[np.array]): list of inputs of length batch.
                Each input is an array of shape (input_dim).

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        inputs = np.stack(inputs)
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, 1)
        tensor_inputs = torch.tensor(inputs).float()
        return self._embedder(tensor_inputs)


class ExperienceEmbedder(Embedder):
    """Optionally embeds each of:

        - state s
        - instructions i
        - actions a
        - rewards r
        - done d

    Then passes a single linear layer over their concatenation.
    """

    def __init__(self, state_embedder, instruction_embedder, action_embedder,
                 reward_embedder, done_embedder, embed_dim):
        """Constructs.

        Args:
            state_embedder (Embedder | None)
            instruction_embedder (Embedder | None)
            action_embedder (Embedder | None)
            reward_embedder (Embedder | None)
            done_embedder (Embedder | None)
            embed_dim (int): dimension of the output
        """
        super().__init__(embed_dim)

        self._embedders = collections.OrderedDict()
        if state_embedder is not None:
            self._embedders["state"] = state_embedder
        if instruction_embedder is not None:
            self._embedders["instruction"] = instruction_embedder
        if action_embedder is not None:
            self._embedders["action"] = action_embedder
        if reward_embedder is not None:
            self._embedders["reward"] = reward_embedder
        if done_embedder is not None:
            self._embedders["done"] = done_embedder

        # Register the embedders so they get gradients
        self._register_embedders = nn.ModuleList(self._embedders.values())
        self._final_layer = nn.Linear(
            sum(embedder.embed_dim for embedder in self._embedders.values()),
            embed_dim)

    def forward(self, instruction_states):
        """Embeds the components for which this has embedders.

        Args:
            instruction_states (list[InstructionState]): batch of states.

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        def get_inputs(key, states):
            if key == "state":
                return [state.observation for state in states]
            elif key == "instruction":
                return [torch.tensor(state.instructions) for state in states]
            elif key == "action":
                actions = np.array(
                        [state.prev_action
                         if state.prev_action is not None else -1
                         for state in states])
                return actions + 1
            elif key == "reward":
                return [state.prev_reward for state in states]
            elif key == "done":
                return [state.done for state in states]
            else:
                raise ValueError(f"Unsupported key: {key}")

        embeddings = []
        for key, embedder in self._embedders.items():
            inputs = get_inputs(key, instruction_states)
            embeddings.append(embedder(inputs))
        return self._final_layer(F.relu(torch.cat(embeddings, -1)))

