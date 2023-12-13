import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from utils import layer_init


class Agent_Memory(nn.Module):
    """
    When RNN policy is enabled, critic and actor share the first several input/preception layers.

    Please find https://github.com/lcswillems/rl-starter-files/tree/e604b36915a13e25ac8a8a912f9a9a15e2d4a170#modelpy
    for more description.
    """

    def __init__(self, envs, hidden_size):
        super().__init__()
        self.shared_network = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.observation_space.shape).prod(), hidden_size)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param.data)
            elif "weight" in name:
                nn.init.xavier_uniform_(param.data)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01),
        )
        self.coverage = nn.Sequential(
            layer_init(nn.Linear(envs.action_space.n, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01),
        )
        self.cw_learner = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.shared_network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(
        self, x, lstm_state, done, action=None, coverage_hist=None
    ):
        hidden, lstm_state = self.get_states(x, lstm_state, done)

        if coverage_hist is None:
            logits = self.actor(hidden)
        else:
            logits = self.actor(hidden)
            c_w = self.cw_learner(hidden)
            c_out = self.coverage(coverage_hist)
            logits = logits + c_w * c_out

        action_probs = Categorical(logits=logits)
        if action is None:
            action = action_probs.sample()
            return (
                action,
                action_probs.log_prob(action),
                action_probs.entropy(),
                self.critic(hidden),
                lstm_state,
            )
        else:
            return (
                action_probs.probs,
                action_probs.log_prob(action),
                action_probs.entropy(),
                self.critic(hidden),
                lstm_state,
            )
