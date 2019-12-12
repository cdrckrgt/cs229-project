'''
Cedrick Argueta

greedy solution
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

import sys
sys.path.append('..')

from PyFEBOLEnv import PyFEBOLEnv
from PyFEBOL import util

run = 'run_02' # run we are comparing to
with open('../../weights/{}.yaml'.format(run), 'r+') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

config['action_dimensionality'] = 'discrete'

env = PyFEBOLEnv(**config)
obs = env.reset()

def select_action():
    best_action = 0
    best_reward = float('-inf')
    for action in range(env.nb_actions):
        state = deepcopy(env)
        _, reward, _, _ = state.step(action)
        if reward >= best_reward:
            best_action = action
            best_reward = reward
    return best_action 

def select_action_exp():
    best_action = 0
    best_reward = float('-inf')
    orig_state = deepcopy(env)
    orig_state.f._predictParticles(config['nb_act_repeat']) # this is the stochastic part
    for action in range(orig_state.nb_actions):
        tuple_action = orig_state.action_space[action]
        score = 0.0
        for obs in [0, 1]:
            state = deepcopy(orig_state)

            state.d.act(tuple_action, state.nb_act_repeat)

            # all of these should be deterministic
            state.f._updateParticles(state.d.getPose(), obs)
            # state.f._resampleParticles()
            state.f._updateBelief()

            reward = state.c.getCost(state.m, state.d, state.f, tuple_action)
            x_particles, y_particles = state.f.x_particles, state.f.y_particles

            prob = state.s.prob(np.asarray([x_particles, y_particles]), state.d.getPose(), obs)


            score += reward * prob.sum() / len(prob)

        if score > best_reward:
            best_action = action
            best_reward = score

    return best_action

render = False
nb_sims = 5
total_mean_tracking_error = 0
total_near_collision_rate = 0
total_avg_reward = 0
for _ in range(nb_sims):
    ep_reward = 0.0
    reward = 0.0
    obs = env.reset()
    for t in count():
        # Select and perform an action
        if render: env.render(mode='save', path='../../gifs/{0}.greedy.frame{1:03d}.png'.format(run, t), freq=1, title='{} greedy solution'.format(run), ep_reward=ep_reward, reward=reward)
        action = select_action_exp()
        _, reward, done, _ = env.step(action)

        ep_reward += reward

        if done:
            mean_tracking_error, near_collision_rate = env.getStats()
            total_mean_tracking_error += mean_tracking_error
            total_near_collision_rate += near_collision_rate
            print('mean tracking error, near collision rate: ', env.getStats())
            if render:
                env.render(mode='save', path='../../gifs/{0}.greedy.frame{1:03d}.png'.format(run, t + 1), freq=1, title='{} greedy solution'.format(run), ep_reward=ep_reward, reward=reward)
                # convert to mp4 and remove frames
                os.system('mkdir ../../gifs/{}'.format(run)) # if it doesn't exist
                os.system('ffmpeg -framerate 8 -i ../../gifs/{0}.greedy.frame%03d.png -c:v libx264 -r 30 ../../gifs/{0}/greedy.mp4'.format(run))
                os.system('rm ../../gifs/{0}.greedy.*.png'.format(run))
            break
    print('reward attained: ', ep_reward)
    print('average cost: ', ep_reward / config['nb_max_episode_steps'])
    total_avg_reward += ep_reward / config['nb_max_episode_steps']
print('mean tracking error: ', total_mean_tracking_error / nb_sims)
print('near collision rate: ', total_near_collision_rate / nb_sims)
print('average cost: ', total_avg_reward / nb_sims)

