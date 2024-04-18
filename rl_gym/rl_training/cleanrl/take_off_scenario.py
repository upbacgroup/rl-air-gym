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
from torch.utils.tensorboard import SummaryWriter


from rl_gym.envs import *
from rl_gym.utils import task_registry
from rl_gym.rl_training.agent.ddpg_agent import Agent
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

if __name__ == "__main__":
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

    agent = Agent(alpha=0.02e-4, beta=0.02e-4, tau=0.001, envs=envs, gamma=0.99,
              max_size=300000, layer1_size=128, layer2_size=128, batch_size=16, num_envs=args.num_envs)
    n_games = 100

    best_score = -10000.0
    score_history = [] #T.tensor([], dtype=T.float32).to(device)
    reward_history = []
    training_data = []
    load_checkpoint = False
    evaluate = False
    # Learning with number of games
    for i in range(n_games):
        observation, _= envs.reset()
        done = T.zeros(args.num_envs, dtype=T.float32).to(device)
        reset = T.zeros(args.num_envs, dtype=T.float32).to(device)
        truncated = False
        score = 0
        while not (done.all() or reset.all()):
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

    if not load_checkpoint:
        figure_file = 'tmp/ddpg/drone.png'
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
    plt.plot(score_history)
    plt.show()
   
    writer.close()