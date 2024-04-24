# Copyright (c) 2024, Regelungs- und Automatisierungstechnik (RAT) - Paderborn University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time

import gym
import isaacgym
from isaacgym import gymutil
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from rl_gym.envs import *
from rl_gym.utils import task_registry, Logger
from rl_gym.rl_training.agent.utils import plot_learning_curve
import matplotlib.pyplot as plt


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "drone", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},

        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},

        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},

        # Algorithm specific arguments
        {"name": "--total-timesteps", "type":int, "default": 30000000,
            "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 0.0026,
            "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type":int, "default": 3, #16 batch_size
            "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--anneal-lr", "action": "store_true", "default": False,
            "help": "Toggle learning rate annealing for policy and value networks"},
        {"name": "--gamma", "type":float, "default": 0.99,
            "help": "the discount factor gamma"},
        {"name": "--gae-lambda", "type":float, "default": 0.95,
            "help": "the lambda for the general advantage estimation"},
        {"name": "--num-minibatches", "type":int, "default": 2,
            "help": "the number of mini-batches"},
        {"name": "--update-epochs", "type":int, "default": 4,
            "help": "the K epochs to update the policy"},
        {"name": "--norm-adv-off", "action": "store_true", "default": False,
            "help": "Toggles advantages normalization"},
        {"name": "--clip-coef", "type":float, "default": 0.2,
            "help": "the surrogate clipping coefficient"},
        {"name": "--clip-vloss", "action": "store_true", "default": False,
            "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
        {"name": "--ent-coef", "type":float, "default": 0.0,
            "help": "coefficient of the entropy"},
        {"name": "--vf-coef", "type":float, "default": 2,
            "help": "coefficient of the value function"},
        {"name": "--max-grad-norm", "type":float, "default": 1,
            "help": "the maximum norm for the gradient clipping"},
        {"name": "--target-kl", "type":float, "default": None,
            "help": "the target KL divergence threshold"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off
    # args.headless = True
    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.device = 'cuda:0'
        # self.num_envs = num_envs
        self.num_obs = input_shape
        self.num_actions = n_actions

        # ALGO Logic: Storage setup
        self.state_memory = T.zeros((self.mem_size, *self.num_obs), dtype=T.float32).to(self.device)
        self.new_state_memory = T.zeros((self.mem_size, *self.num_obs), dtype=T.float32).to(self.device)
        self.action_memory = T.zeros((self.mem_size, self.num_actions), dtype=T.float32).to(self.device)
        self.reward_memory = T.zeros((self.mem_size), dtype=T.float32).to(self.device)
        self.terminal_memory = T.zeros((self.mem_size), dtype=T.float32).to(self.device)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = T.randint(0, max_mem, (batch_size,), device=self.device)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 1e-3
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), weight_decay= 1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = T.tanh(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 1e-3
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=4, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='Actor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic')

    def choose_action(self, observation):
        mu = self.actor.forward(observation)
        return mu
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print("..........Loading model...........")
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        print("..........Loading model done...........")
    

def play(args, agent, env, env_cfg):
    
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = 800 # number of steps before plotting states
    counter = 0

    observation, _ = env.reset()
    for i in range(1000*int(env.max_episode_length)):
        if counter == 0:
            start_time = time.time()
        counter += 1
        states = observation.clone()

        actions = agent.choose_action(states)
        obs, _, rewards, *_ = env.step(actions.detach())
        observation = obs.clone()



if __name__ == "__main__":
    args = get_args()
    
    # # env setup
    env_cfg = task_registry.get_cfgs(name="drone")
    
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.control.controller = "lee_position_control"

    # prepare environment
    env, *_ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    print("Arg.task: ", args.task)
    # print("using device:", device)
    agent = Agent(alpha=1e-4, beta=1e-3, input_dims=[13], tau=0.001, gamma= 0.99, n_actions=4, max_size=1000000, env=env,
            layer1_size=400, layer2_size=300, batch_size=64)
    agent.load_models()
    play(args, agent, env, env_cfg)    