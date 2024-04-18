# Copyright (c) 2024, Regelungs- und Automatisierungstechnik (RAT) - Paderborn University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from rl_gym.envs import *
from rl_gym.utils import task_registry
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

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
 
class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = T.zeros(self.num_envs, dtype=T.float32, device=self.device)
        self.episode_lengths = T.zeros(self.num_envs, dtype=T.int32, device=self.device)
        self.returned_episode_returns = T.zeros(self.num_envs, dtype=T.float32, device=self.device)
        self.returned_episode_lengths = T.zeros(self.num_envs, dtype=T.int32, device=self.device)

        return observations

    def step(self, action):
        observations, privileged_observations, rewards, resets, infos, dones = super().step(action)
        
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - resets
        self.episode_lengths *= 1 - resets
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
            resets,
        )

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.action_upper_limits = T.tensor(
            [1, 1, 1, np.pi], device=self.device, dtype=T.float32)
        self.action_lower_limits = T.tensor(
            [-1, -1, 0.0, -np.pi], device=self.device, dtype=T.float32)

        self.checkpoint_file = os.path.join(chkpt_dir, name + '.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 3e-3 #1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 3e-3
        # f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = T.tanh(self.mu(x))
        # x = F.softmax(self.mu(x), dim=1)
        x.clamp(self.action_lower_limits, self.action_upper_limits)
        return x

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
        self.q_upper_limits = T.tensor(
            [0], device=self.device, dtype=T.float32)
        self.q_lower_limits = T.tensor(
            [-100.0], device=self.device, dtype=T.float32)

        self.checkpoint_file = os.path.join(chkpt_dir, name + '.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 0.003 #1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        f2 = 0.003 #1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)


        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        # T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        # T.nn.init.uniform_(self.q.bias.data, -f3, f3)
    

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
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

        # return state_action_value.clamp(self.q_lower_limits, self.q_upper_limits)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, alpha, beta, tau, envs, gamma=0.99,
                 max_size=100000, layer1_size=400,
                 layer2_size=300, batch_size=64, num_envs = 1):
        self.gamma = gamma
        self.tau=tau
        self.device_type = 'cuda:0'
        self.num_envs = num_envs
        self.batch_size = batch_size
        input_dims = [3] #[envs.num_obs]
        self.n_actions = envs.num_actions
        self.memory=ReplayBuffer(max_size, envs.num_obs, envs.num_actions, self.device_type, self.num_envs)

        

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=self.n_actions,
                                  name='Actor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=self.n_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=self.n_actions,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, n_actions=self.n_actions,
                                           name='TargetCritic')
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))
        self.update_network_parameters(tau=1)
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        self.actor.eval()
        mu = self.actor.forward(observation)
        
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        
        self.actor.train()
        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_memory(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_target = self.target_critic.forward(new_state, target_actions)
        critic_value_t = self.critic.forward(state, action).detach()
        
        target = T.zeros_like(reward, dtype=T.float32).to(self.critic.device)
        
        # squeeze the tensor to keep compatible dims
        critic_value_ = T.squeeze(critic_value_target, dim=2)
        critic_value = T.squeeze(critic_value_t, dim=2)

        # bootstrap value if not done
        for j in range(self.batch_size):
            target[j] = reward[j] + self.gamma*critic_value_[j]*(1-done[j])

        self.critic.eval()
        self.critic.optimizer.zero_grad()
        self.critic.train()
        critic_loss = F.mse_loss(target, critic_value) #TODO: Check here!
        critic_loss.backward()
        self.critic.optimizer.step()
 
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)

        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

if __name__ == "__main__":
    T.cuda.empty_cache()
    # import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    args = get_args()
    

    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    T.backends.cudnn.deterministic = args.torch_deterministic

    device = args.sim_device
    print("using device:", device)

    # env setup
    envs, env_cfg = task_registry.make_env(name="drone", args=args)

    envs = RecordEpisodeStatisticsTorch(envs, device)

    print("num actions: ",envs.num_actions)
    print("num obs: ", envs.num_obs)
    print("num envs:", args.num_envs)

    agent = Agent(alpha=0.05e-1, beta=0.05e-1, tau=0.001, envs=envs, gamma=0.99,
              max_size=100000, layer1_size=400, layer2_size=300, batch_size=16, num_envs=args.num_envs)
    n_games = 100

    best_score = -10000.0
    score_history = [] #T.tensor([], dtype=T.float32).to(device)
    reward_history = []
    training_data = []
    load_checkpoint = False
    evaluate = False
    # Learning with number of games
    for i in range(n_games):
        # Observe the environment
        observation, _= envs.reset()
        done = T.zeros(args.num_envs, dtype=T.float32).to(device)
        reset = T.zeros(args.num_envs, dtype=T.float32).to(device)
        truncated = False
        score = 0
        # print("done tensor: ", done)
        # states = observation
        while not (reset or done):
            # states = observation
            action = agent.choose_action(observation[:, :3]) #observation[:, :3]
            new_states, reward, done, info, reset= envs.step(action)
            print(f'done: {done}, reset: {reset}')
            agent.remember(observation[:, :3], action, reward, new_states[:, :3], done)
            agent.learn()
            score += reward
            observation = new_states
        score_history.append(score.cpu().numpy())
        avg_score = np.mean(score_history[-100:])


        if avg_score > best_score:
            print(".........Saving model...........")
            agent.save_models()
            best_score = avg_score

        print(f'episode {i} score {score} avg score {avg_score}')
        # if done.all() or reset.all():
        #     print(".........Saving model...........")
        #     agent.save_models()

    if not load_checkpoint:
        figure_file = 'tmp/ddpg/drone.png'
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
    plt.plot(score_history)
    plt.show()
   
    writer.close()