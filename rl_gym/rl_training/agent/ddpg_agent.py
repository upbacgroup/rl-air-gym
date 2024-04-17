from .neural_network_model import ActorNetwork, CriticNetwork
from .utils import ReplayBuffer, OUActionNoise
import torch as T
import numpy as np
import torch.nn.functional as F

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