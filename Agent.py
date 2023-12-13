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
            policy_clip=0.2, batch_size=64, n_epochs=10, DrQ = False):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.device = 'cuda'
        self.transform = K.RandomCrop(size=self.env.observation_space.shape,
                                      padding=(8, 8),
                                      padding_mode='replicate').to('cuda') if DrQ else None

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
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    
    def choose_action(self, obs):
        state = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device)

        self.standardize(state)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                states = self._preprocess(states)
                
                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()  




    # def learn(self):  #learn de jolou nais para referencia
    #     obs, act, rew, obs_next, done = self.replay_buffer.sample(self.batch_size)
    #     with torch.no_grad():
    #         self.model.reset_noise()
    #         self.target_model.reset_noise()
    #         act = torch.as_tensor(act,
    #                               dtype=torch.int64,
    #                               device=self.device)
    #         rew = torch.as_tensor(rew,
    #                               dtype=torch.float32,
    #                               device=self.device)
    #         obs_next = self._preprocess(obs_next)
    #         done = torch.as_tensor(done,
    #                                dtype=torch.float32,
    #                                device=self.device)

    #         target_q = self.target_model(obs_next).detach()
    #         if self.is_double:
    #             max_act = self.model(obs_next).detach()
    #             max_act = torch.argmax(max_act, dim=1)
    #             length = self.batch_size * 2 if self.transform else self.batch_size
    #             max_target_q = target_q[torch.arange(length), max_act]
    #             max_target_q = max_target_q.unsqueeze(-1)
    #         else:
    #             max_target_q, _ = target_q.max(dim=1, keepdims=True)
    #         if self.transform:
    #             max_target_q = max_target_q[:self.batch_size] + max_target_q[self.batch_size:]
    #             max_target_q /= 2.
    #         target = rew + self.gamma * max_target_q * (1. - done)

    #     obs = self._preprocess(obs)

    #     self.model.train()
    #     self.optimizer.zero_grad(set_to_none=True)
    #     q = self.model(obs)
    #     if self.transform:
    #         q = (q[:self.batch_size] + q[self.batch_size:]) / 2.
    #     q = torch.gather(q, 1, act)
    #     loss = self.criterion(q, target)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
    #     self.optimizer.step()
    #     self.model.eval()

    #     with torch.no_grad():
    #         loss = float(loss.detach().cpu().numpy())
    #         if self.target_replace_steps % self.target_steps == 0:
    #             self.target_model.load_state_dict(self.model.state_dict())
    #             self.target_model.eval()
    #             print('target replaced')
    #         self.target_replace_steps += 1
    #     return loss
