'''
Cedrick Argueta

demo
'''


import gym

import random
import math
import numpy as np
from itertools import count
import yaml
from copy import deepcopy
import os
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from PyFEBOLEnv import PyFEBOLEnv
from PyFEBOL import util

run = 'run_02' # run we are comparing to
with open('../../weights/{}.yaml'.format(run), 'r+') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

config['sensor_sd'] = 25.0
config['action_dimensionality'] = 'continuous'

env = PyFEBOLEnv(**config)
obs = env.reset()

def select_action():
    while True:
        try:
            control = input('Use wasd to enter a direction!')
            if len(control) == 1 and control in 'wasd':
                return control
            print('Choose a valid command please...')
        except EOFError:
            os._exit(110)


plt.ion()

ep_reward = 0.0
reward = 0.0
obs = env.reset()
for t in count():
    # Select and perform an action
    env.render(mode='live', freq=1, title='{} demo'.format(run), ep_reward=ep_reward, reward=reward)
    action = select_action()
    plt.close()
    if action == 'w':
        action = (np.pi / 2, 0)
    elif action == 'a':
        action = (np.pi, 0)
    elif action == 's':
        action = (3 * np.pi / 2, 0)
    else:
        action = (2 * np.pi, 0)
    _, reward, done, _ = env.step(action)

    ep_reward += reward

    if done:
        mean_tracking_error, near_collision_rate = env.getStats()
        print('mean tracking error, near collision rate: ', env.getStats())

        env.render(mode='live', freq=1, title='{} demo'.format(run), ep_reward=ep_reward, reward=reward)
        break

print('reward attained: ', ep_reward)
print('average cost: ', ep_reward / config['nb_max_episode_steps'])
