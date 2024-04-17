# Copyright (c) 2024, Regelungs- und Automatisierungstechnik (RAT) - Paderborn University
# All rights reserved.

import numpy as np
import torch as T
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, max_size, num_obs, num_actions, device='cuda:0', num_envs=1):
        self.max_size = max_size
        self.num_envs = num_envs
        self.num_obs = 3 #num_obs
        self.num_actions = num_actions
        self.device = device
        self.mem_cntr = 0
        # ALGO Logic: Storage setup
        self.state_memory = T.zeros((self.max_size, self.num_envs, self.num_obs), dtype=T.float32).to(self.device)
        self.new_sate_memory = T.zeros((self.max_size, self.num_envs, self.num_obs), dtype=T.float32).to(self.device)
        self.action_memory = T.zeros((self.max_size, self.num_envs, self.num_actions), dtype=T.float32).to(self.device)
        self.reward_memory = T.zeros((self.max_size, self.num_envs), dtype=T.float32).to(self.device)
        self.done_memory = T.zeros((self.max_size, self.num_envs), dtype=T.float32).to(self.device)


    def store_memory(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.max_size
        self.state_memory[index] = state
        self.new_sate_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.max_size)

        # Generate random indices
        indices = T.randint(0, max_mem, (batch_size,), device=self.device)

        # Retrieve data using the random indices
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        new_states = self.new_sate_memory[indices]
        terminals = self.done_memory[indices]

        return states, actions, rewards, new_states, terminals
    


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-3, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)
    


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)