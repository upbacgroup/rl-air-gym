# Copyright (c) 2024, Regelungs- und Automatisierungstechnik (RAT) - Paderborn University
# All rights reserved.

import os
import random
import time

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.checkpoint_file = os.path.join(chkpt_dir, name + '.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(3e-3)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.checkpoint_file = os.path.join(chkpt_dir, name + '.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims+self.n_actions, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)
        self.relu = nn.ReLU()
        self.init_weights(3e-3)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(self.device)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x, a):
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(T.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))