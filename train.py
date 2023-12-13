import gym
import torch
from torch.backends import cudnn
from utils import plot_learning_curve
import numpy as np
from collections import deque
import time

from Agent import Agent
import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

##chpt_dir es la carpeta donde se guardan los pesos de los modelos
def get_models(env: gym.Env, chkpt_dir, n_frames: int):
    actor = models.SimpleExtractor(env.observation_space.shape, n_frames)
    actor = models.ActorNetwork(actor,env.action_space.n, actor.units,0.0003, chkpt_dir = chkpt_dir)
    critic = models.SimpleExtractor(env.observation_space.shape, n_frames)
    critic = models.CriticNetwork(critic, critic.units, 0.0003, chkpt_dir = chkpt_dir)
    #m = models.DuelingMLP(m, env.action_space.n, noisy=False)
    return actor.to(DEVICE), critic.to(DEVICE)


def train(dqn):
    print('training started')
    dqn.save_explorations(60)
    dqn.load_explorations()
    # raise ValueError
    dqn.learn()  # warmup

    saved_rew = float('-inf')
    saved_train_rew = float('-inf')
    for i in range(300):
        print('episode', i + 1)
        rew, loss = dqn.run_episode()
        if rew > saved_train_rew:
            print('new best train model found')
            saved_train_rew = rew
            dqn.save_models('besttrain')
        if i % 10 == 0:
            eval_rew = dqn.evaluate()
            if eval_rew > saved_rew:
                print('new best eval model found')
                saved_rew = eval_rew
                dqn.save_models('best')
        dqn.save_models('latest')

        dqn.log({'reward': rew, 'loss': loss})
        print(f'episode {i + 1} finished, total step {dqn.steps}, epsilon {dqn.eps}',
              f'total rewards {rew}, loss {loss}', sep='\n')
        print()

        

if __name__ == '__main__':
    n_frames = 4
    env = hkenv.HKEnv((192, 192), w1=0.8, w2=0.8, w3=-0.0001)
    actor_network, critic_network = get_models(env,"tmp/ppo_simplified" ,n_frames)
    # actor_network.load_state_dict(torch.load(f'tmp/pppo2/actor_torch_ppo.pth')) 
    # critic_network.load_state_dict(torch.load(f'tmp/pppo2/critic_torch_ppo.pth')) 
    N = 100
    batch_size = 32
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape, actor=actor_network, critic=critic_network)
    n_games = 500

    figure_file = 'plots/HK.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        stacked_obs = deque(
            (observation for _ in range(n_frames)),
            maxlen=n_frames
        )
        done = False
        score = 0
        while not done:
            obs_tuple = tuple(stacked_obs)
            model_input = np.array([obs_tuple], dtype=np.float32)
            action, prob, val = agent.choose_action(model_input)
            observation_, reward, done, info, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(obs_tuple, action, prob, val, reward, done)
            if n_steps % N == 0:
                #print("Aprendiendo :>")
                
                #env.pause()
                agent.learn()
                #env.pause()
                
                learn_iters += 1
            observation = observation_
            stacked_obs.append(observation)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)