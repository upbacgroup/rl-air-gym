import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(p.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 400)
        self.fc2 = nn.Linear(400 + np.prod(env.single_action_space.shape), 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        x = T.cat([x, a], 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", T.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=T.float32)
        )
        self.register_buffer(
            "action_bias", T.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=T.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias