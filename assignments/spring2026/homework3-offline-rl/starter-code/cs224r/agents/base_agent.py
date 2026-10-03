class BaseAgent:
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.learning_freq = agent_params['learning_freq']
        self.optimizer_spec = agent_params['optimizer_spec']
        self.replay_buffer = None
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []
