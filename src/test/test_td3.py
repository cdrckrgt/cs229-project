'''
Cedrick Argueta

majority of code taken from:
https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
import numpy as np
from itertools import count
import os
import yaml

import sys
sys.path.append('..')

from PyFEBOLEnv import PyFEBOLEnv

'''
configuration
-------------

code in this section does multiple things for setting up evaluation of an agent.

things to note:
    mostly follows the same format as train_dqn.py

    the yaml file loaded determines which model is being evaluated.

    a hidden dependency that isn't stated elsewhere: ffmpeg to create gifs
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on: ", device)

run = 'td3_run_00'
ep = '.ep3500'
with open('../../weights/{}.yaml'.format(run), 'r+') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

assert config['action_dimensionality'] == 'continuous', 'You must use a run trained with TD3 with this tester!'
phi_length = config['phi_length']
nb_act_repeat = config['nb_act_repeat']

env = PyFEBOLEnv(**config)
obs = env.reset()

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_fc1 = nn.Linear(7, 512)
        self.obs_act1 = nn.ReLU()
    
        self.obs_fc2 = nn.Linear(512, 384)
        self.obs_act2 = nn.ReLU()

        # two actions: degree for translation and rotation
        self.obs_fc3 = nn.Linear(384, 1)
        self.obs_act3 = nn.Tanh()

    def forward(self, obs, hidden=None):

        obs = self.obs_act1(self.obs_fc1(obs))
        obs = self.obs_act2(self.obs_fc2(obs))
        actions = self.obs_act3(self.obs_fc3(obs))

        return actions

# actor: from beliefs and poses to actions
mu = Actor().double().to(device)
mu.load_state_dict(torch.load('../../weights/{}.mu.pt'.format(run + ep), map_location=device))
mu.eval()

def select_action(state):
    with torch.no_grad():
        action = mu(torch.from_numpy(state).to(device))
        action = action.cpu().numpy()
        action = action.clip(-1, 1)
        action *= np.pi
        assert action >= -np.pi and action <= np.pi, 'action out of bounds: {}'.format(action)
        return action.item()

'''
testing
-------

runs a number of test episodes and creates gifs of the results.
'''

total_mean_tracking_error = 0
total_near_collision_rate = 0
total_avg_reward = 0
render = False

nb_sims = 20
print("beginning testing...")
for i_episode in range(nb_sims):
    print("beginning episode {}...".format(i_episode))

    obs = env.reset()
    obs = obs[np.newaxis, :]

    ep_reward = 0.
    reward = 0.

    for t in count():
        # Select and perform an action
        if render: env.render(mode='save', path='../../gifs/{0}.test{1:02d}.frame{2:03d}.png'.format(run, i_episode, t), freq=1, title='{} TD3 solution'.format(run), ep_reward=ep_reward, reward=reward)
        action = select_action(obs)
        action_tuple = (action, 0) # heading angle is ineffectual for bearing only sensor

        next_obs, reward, done, _ = env.step(action_tuple)
        next_obs = next_obs[np.newaxis, :]

        reward = reward * nb_act_repeat 
        ep_reward += reward

        obs = next_obs

        if done:
            mean_tracking_error, near_collision_rate = env.getStats()
            total_mean_tracking_error += mean_tracking_error
            total_near_collision_rate += near_collision_rate
            print('mean tracking error, near collision rate: ', env.getStats())

            if render:
                env.render(mode='save', path='../../gifs/{0}.test{1:02d}.frame{2:03d}.png'.format(run, i_episode, t + 1), freq=1, title='{} TD3 solution'.format(run), ep_reward=ep_reward, reward=reward)
                # convert to mp4 and remove frames
                os.system('mkdir -p ../../gifs/{}'.format(run)) # if it doesn't exist
                os.system('ffmpeg -framerate 8 -i ../../gifs/{0}.test{1:02d}.frame%03d.png -c:v libx264 -r 30 ../../gifs/{0}/test{1:02d}.mp4'.format(run, i_episode))
                os.system('rm ../../gifs/{0}.test{1:02d}.*.png'.format(run, i_episode))
            break

    print('reward attained: ', ep_reward)
    print('average cost: ', ep_reward / config['nb_max_episode_steps'])
    total_avg_reward += ep_reward / config['nb_max_episode_steps']
print('mean tracking error: ', total_mean_tracking_error / nb_sims)
print('near collision rate: ', total_near_collision_rate / nb_sims)
print('average cost: ', total_avg_reward / nb_sims)

print('complete')
