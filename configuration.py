import numpy as np
import model
import torch.nn.functional as F


class Configuration():
    def __init__(self):

        # random seed
        self.random_seed = 40

        # configs for replay buffer
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 128  # minibatch size
        
        # configs for agent
        self.gamma = 0.99  # discount factor
        self.tau = 1e-3  # for soft update of target parameters
        self.learn_frequency = 1  # learn every `learn_frequency` timesteps
        self.num_experience_replays = 2  # how many times to learn in each learning period
        
        # configs for actor
        self.actor = model.Actor
        self.lr_actor = 1e-4  # learning rate of the actor
        self.batch_normalization_actor = True
        
        # configs for critic
        self.critic = model.Critic
        self.lr_critic = 1e-3  # learning rate of the critic
        self.l2_weight_decay = 0  # L2 weight decay
        self.batch_normalization_critic = True

        # configs for noise sample
        self.mu = 0.
        self.theta = 0.15
        self.sigma = 0.2
        self.noise_func = np.random.standard_normal
