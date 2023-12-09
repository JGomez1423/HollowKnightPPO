import os
import copy
import time
import torch
import random
import numpy as np
from torch import nn
from collections import deque
from kornia import augmentation as K
from torch.utils.tensorboard import SummaryWriter

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Agent:
    def __init__ (self, n_actions, input_dims, actor: nn.Module, critic: nn.Module, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = actor
        self.critic = critic


        self.memory = PPOMemory(batch_size)
        
    @staticmethod
    def standardize(obs):
        # values found from empirical data
        obs -= 92.54949702814011
        obs /= 57.94090462506912
        return obs
        
    def _preprocess(self, obs):
        if len(obs.shape) < 4:  # not image
            return torch.as_tensor(obs, dtype=torch.float32,
                                   device=self.device).squeeze()
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device)
        self.standardize(obs)
        if self.transform:
            scale = torch.randn((self.batch_size, 1, 1, 1),
                                dtype=torch.float32, device=self.device)
            scale = torch.clip(scale, -2, 2) * 0.025 + 1.
            return torch.vstack((obs * scale, self.transform(obs)))
        else:
            return obs