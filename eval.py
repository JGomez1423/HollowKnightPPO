import os
import random
import time
import gym
import torch
import numpy as np
from collections import deque
from torch.backends import cudnn

import hkenv
import models
import trainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


def get_model(env: gym.Env, n_frames: int):
    actor = models.SimpleExtractor(env.observation_space.shape, n_frames)
    actor = models.ActorNetwork(actor,env.action_space.n, actor.units,0.0003)
    return actor.to(DEVICE)


@staticmethod
def standardize(obs):
    # values found from empirical data
    obs -= 92.54949702814011
    obs /= 57.94090462506912
    return obs

def choose_action(obs, actor):
    state = torch.as_tensor(obs, dtype=torch.float32,
                              device='cuda')

    standardize(state)
    dist = actor(state)
    action = dist.sample()
    action = torch.squeeze(action).item()
    return action



@torch.no_grad()
def main():
    n_frames = 4
    env = hkenv.HKEnv((192, 192), w1=0.8, w2=0.8, w3=0.0001)
    actor_network = get_model(env, n_frames)
    actor_network.eval()
    print("Modelo cargado exitosamente.")
    actor_network.load_state_dict(torch.load(f'tmp/ppo/actor_torch_ppo.pth'))  # replace this path with your weight file
    n_games = 5
    for i in range(n_games):
        initial, _ = env.reset()
        stacked_obs = deque(
            (initial for _ in range(n_frames * 2 - 1)),
            maxlen=n_frames
        )

        while True:
            t = time.time()
            obs_tuple = tuple(stacked_obs)
            
            obs = np.array([obs_tuple], dtype=np.float32)
            #print("Dimensiones de entrada:", obs.shape)
            action = choose_action(obs, actor_network)
            obs_next, rew, done, _, _ = env.step(action)
            print(action, rew)
            stacked_obs.append(obs_next)
            if done:
                break
            t = 0.16 - (time.time() - t)
            
            if t > 0:
                time.sleep(t)


if __name__ == '__main__':
    main()
